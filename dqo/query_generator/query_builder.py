from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass
from typing import cast, Set, List, Callable, Dict, Optional, Tuple

import numpy as np

from dqo.db.models import Database
from dqo.query_generator import rand_condition
from dqo.relational import Query
from dqo.relational.models import Condition, Projection, Join, TableRef


@dataclass
class ConditionGenerator:
    schema: Database

    def generate(self, query: Query) -> Optional[Condition]:
        if not any(query._relations):
            return None

        relation: TableRef = np.random.choice(sorted(list(query.relations)))
        table = self.schema[relation.name]
        column = np.random.choice(sorted(table.columns))

        return rand_condition(column)


class ValidQueryParts:
    projections: Set[Projection]
    joins: Set[Join]
    relations: Set[TableRef]

    generate_condition: Callable[[Query], Condition]

    def __init__(self, schema: Database = None):
        self.schema = schema
        if schema:
            # Fixed parts
            self.projections = set(Projection(c.to_ref(), 'MIN') for c in schema.columns)
            self.joins = self.valid_joins(schema)
            self.relations = set(t.to_ref() for t in schema.tables)
            # TODO: conditions should not be bound, the same column can appear many times with real perf effect
            self.generate_condition = ConditionGenerator(schema).generate

    @staticmethod
    def valid_joins(schema: Database) -> Set[Join]:
        result = set()
        tables = schema.tables[:]
        left = tables.pop()

        for t in left.types_lookup.keys():
            for lc in left.types_lookup[t]:
                for right in tables:
                    if t in right.types_lookup:
                        for rc in right.types_lookup[t]:
                            result.add(Join(lc.to_ref(), rc.to_ref(), '='))

        return result

    def __copy__(self) -> 'ValidQueryParts':
        cpy = ValidQueryParts()
        cpy.projections = self.projections.copy()
        cpy.joins = self.joins.copy()
        cpy.generate_condition = self.generate_condition
        cpy.relations = self.relations.copy()

        return cpy

    def copy(self):
        return self.__copy__()


class MutationType(enum.IntEnum):
    Add = 0
    Remove = 1


@dataclass
class Mutation:
    name: str
    do: Callable
    validate: Callable
    type: MutationType


class QueryBuilder:
    q: Query
    prev_q: Query

    valid_query_parts: ValidQueryParts
    prev_valid_query_parts: ValidQueryParts

    mutations: List[Mutation]
    seed: Optional[int] = None

    def __init__(self, schema: Database, seed: Optional[int] = None):
        # TODO: this seems odd
        self._valid_query_parts = ValidQueryParts(schema)

        self.valid_query_parts = self._valid_query_parts.copy()
        self.mutations = [
            Mutation(do=self.add_projection, validate=self.can_add_projection, name='add projection', type=MutationType.Add),
            Mutation(do=self.add_condition, validate=self.can_add_condition, name='add condition', type=MutationType.Add),
            Mutation(do=self.add_relation, validate=self.can_add_relation, name='add relation', type=MutationType.Add),
            Mutation(do=self.remove_projection, validate=self.can_remove_projection, name='remove projection', type=MutationType.Remove),
            Mutation(do=self.remove_condition, validate=self.can_remove_condition, name='remove condition', type=MutationType.Remove),
            Mutation(do=self.remove_relation, validate=self.can_remove_relation, name='remove relation', type=MutationType.Remove)
        ]

        self.q = Query()
        self.seed = seed
        if seed:
            np.random.seed(seed)
        # history
        self.prev_q = None
        self.prev_valid_query_parts = None

    def reset(self):
        self.valid_query_parts = self._valid_query_parts.copy()
        self.q = Query()
        self.prev_q = None
        self.prev_valid_query_parts = None

    def mutate(self, mutation_idx: int) -> Query:
        try:
            mutation = self.mutations[mutation_idx]
            mutation.do()
        except Exception as e:
            print(e)
        return self.q

    def get_available_mutations(self, mtype: MutationType = None) -> List[int]:
        mutations = [
            m for m in self.mutations if m.type == mtype
        ] if mtype is not None else self.mutations[:]

        return [i for i, m in enumerate(mutations) if m.validate()]

    def _save_state(self):
        self.prev_q = self.q.copy()
        self.prev_valid_query_parts = self.valid_query_parts.copy()

    def undo(self):
        if self.prev_q and self.prev_valid_query_parts:
            self.q = self.prev_q.copy()
            self.valid_query_parts = self.prev_valid_query_parts.copy()

        self.prev_q = None
        self.prev_valid_query_parts = None

    def sync(self):
        for p in self.q._projections:
            if p in self.valid_query_parts.projections:
                self.valid_query_parts.projections.remove(p)

        for r in self.q._relations:
            if r in self.valid_query_parts.relations:
                self.valid_query_parts.relations.remove(r)

    def available_projections(self) -> List[Projection]:
        if len(self.valid_query_parts.projections) == 0:
            return []

        if len(self.q._relations) == 0:
            return []

        possible_projections = []

        for c in self.valid_query_parts.projections:
            if c.col.table in self.q._relations:
                possible_projections.append(c)

        return possible_projections

    def add_condition(self):
        condition = self.valid_query_parts.generate_condition(self.q)

        if condition is None:
            return None
        self._save_state()

        self.q.add_condition(condition)

    def remove_condition(self):
        if len(self.q._conditions) == 0:
            return

        # choose one at random (this can be improved)
        c = np.random.choice(sorted(list(self.q._conditions)))

        self._save_state()

        # remove
        cascaded = self.q.remove_condition(c, cascade=True)

        if cascaded:
            self.valid_query_parts.relations.add(c.col.table)

    def add_projection(self):
        relevant_projections = self.available_projections()

        if len(relevant_projections) == 0:
            return

        p = cast(Projection, np.random.choice(sorted(relevant_projections)))

        self._save_state()

        self.valid_query_parts.projections.remove(p)
        self.q.add_projection(projection=p)

    def remove_projection(self):
        if len(self.q._projections) == 0:
            return

        # choose one at random (this can be improved)
        p = np.random.choice(sorted(list(self.q._projections)))

        self._save_state()

        # return back to selectable items
        self.valid_query_parts.projections.add(p)

        # remove
        cascaded = self.q.remove_projection(p, cascade=True)
        if cascaded:
            self.valid_query_parts.relations.add(p.col.table)

    def add_relation(self):
        # If this is the only table, add it using a projection
        if len(self.q._relations) == 0:
            t = cast(TableRef, np.random.choice(sorted(list(self.valid_query_parts.relations))))
            table_projections = [p for p in self.valid_query_parts.projections if p.col.table.alias == t.alias]
            p = cast(Projection, np.random.choice(sorted(table_projections)))
            self.valid_query_parts.projections.remove(p)
            self.q.add_projection(projection=p)
        # else, use a join
        elif len(self.valid_query_parts.joins) > 0:
            relevant_joins = [
                j for j in self.valid_query_parts.joins
                if j.left.table in self.q._relations or j.right.table in self.q._relations
            ]

            if not relevant_joins:
                return

            j = cast(Join, np.random.choice(sorted(relevant_joins)))

            self._save_state()

            self.valid_query_parts.joins.remove(j)
            self.q.add_join(join=j)

    def remove_relation(self):
        if len(self.q._relations) == 0:
            return self.q
        elif len(self.q._relations) == 1:
            for p in list(self.q.projections):
                self.q.remove_projection(p, cascade=True)
            for c in list(self.q._conditions):
                self.q.remove_condition(c, cascade=True)

            return self.q
        else:
            if not self.q._relations:
                return

            relation_joins: Dict[TableRef, List[Join]] = defaultdict(list)
            relation_join_counts: Dict[TableRef, int] = defaultdict(int)

            for j in self.q.joins:
                for side in [j.left, j.right]:
                    relation_joins[side.table].append(j)
                    relation_join_counts[side.table] += 1

            # pick the relation with the least effect on query structure (jenga style)
            selected_relation = min(relation_join_counts, key=relation_join_counts.get)

            self._save_state()

            for i, j in enumerate(relation_joins[selected_relation]):
                other = j.left.table if j.left.table == selected_relation else j.right.table
                for p in list(self.q.projections):
                    if p.col.table == other:
                        self.q.remove_projection(p, cascade=True)
                for c in list(self.q.conditions):
                    if c.col.table == other:
                        self.q.remove_condition(c, cascade=True)

                deleted = self.q.remove_join(j, cascade=True)
                # return back to selectable items
                self.valid_query_parts.joins.add(j)
                if deleted:
                    for d in deleted:
                        self.valid_query_parts.relations.add(d)

    def replace_join(self):
        if not self.q._relations or len(self.q._relations) <= 1:
            return self.q
        else:
            candidates: List[Tuple[Join, Join]] = []
            for existing_join in self.q._joins:
                sides = [existing_join.left, existing_join.right]
                for valid_join in self.valid_query_parts.joins:
                    if valid_join.left in sides and valid_join.right in sides:
                        candidates.append((valid_join, existing_join))
            if not candidates:
                return self.q

            to_add, to_remove = np.random.choice(candidates)

            self.q.add_join(to_add)
            self.q.remove_join(to_remove)

            self.valid_query_parts.joins.remove(to_add)
            self.valid_query_parts.joins.add(to_remove)

    def can_add_relation(self) -> bool:
        return any(self.valid_query_parts.relations)

    def can_add_projection(self) -> bool:
        return any(self.available_projections())

    def can_add_condition(self) -> bool:
        return any(self.q.relations)

    def can_remove_relation(self) -> bool:
        return len(self.q._relations) > 1

    def can_remove_projection(self) -> bool:
        return len(self.q._projections) > 1

    def can_remove_condition(self) -> bool:
        return any(self.q.conditions)

    def can_replace_join(self) -> bool:
        if len(self.q._joins) == 0 or len(self.valid_query_parts.joins) == 0:
            return False

        for existing_join in self.q._joins:
            sides = [existing_join.left, existing_join.right]
            for valid_join in self.valid_query_parts.joins:
                if valid_join.left in sides and valid_join.right in sides:
                    return True

        return False
