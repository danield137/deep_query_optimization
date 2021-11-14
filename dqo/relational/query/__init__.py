from __future__ import annotations

from typing import List, Any, Set, cast, Tuple, Iterator

from dqo.relational.models import Selection, ColumnRef, TableRef, Condition, Join, Projection


class Query:
    """
    Query class is a utility class to help build queries.
    It contains the basic building block for relational queries.
    A query object is a conceptual view over a relational tree, and as such,
     is mapped 1-to-Many (a single query can become many relational trees),
     but two equal queries should always produce the same result set.
    """
    # TODO: this object only supports simple queries. should add support for nested queries.
    _conditions: Set[Condition]
    _projections: Set[Projection]
    _joins: Set[Join]
    _relations: Set[TableRef]

    _joint_tables: Set[Tuple[TableRef, TableRef]]

    def __init__(self, track_order=False):
        self._conditions = set()
        self._projections = set()
        self._joins = set()
        self._relations = set()

        self._joint_tables = set()
        self.__track_order = track_order

        if track_order:
            self.__projections_order = {}
            self.__selections_order = {}
            self.__relations_order = {}

            self.__global_indexer = 0

            # this is the quickest way to handle consistent order when deleting items
            def __next_index():
                self.__global_indexer += 1
                return self.__global_indexer

            self.__next_index = __next_index

    def __len__(self) -> int:
        return sum(len(list(l)) for l in [self._joins, self._projections, self.conditions, self._relations])

    def __hash__(self):
        return hash(self.to_sql(pretty=False, alias=False))

    @property
    def selections(self) -> Iterator[Selection]:
        for c in self.conditions:
            yield cast(Selection, c)

        for j in self.joins:
            yield cast(Selection, j)

    @property
    def joins(self) -> Iterator[Join]:
        for j in self._joins:
            yield j

    @property
    def conditions(self) -> Iterator[Condition]:
        for c in self._conditions:
            yield c

    @property
    def relations(self) -> Iterator[TableRef]:
        for r in self._relations:
            yield r

    @property
    def projections(self) -> Iterator[Projection]:
        for p in self._projections:
            yield p

    def add_table(self, t: TableRef, track: bool = False) -> TableRef:
        # since this can be called from various places, tracking only counts
        # for explicit relation
        if self.__track_order and t not in self.__relations_order and track:
            self.__relations_order[t] = self.__next_index()
        if t not in self._relations:
            self._relations.add(t)

        for r in self._relations:
            if hash(r) == hash(t):
                return r

    def add_join(self, join_cols: List[ColumnRef] = None, join: Join = None, with_relations: bool = False) -> Join:
        join_cols = join_cols or [join.left, join.right]
        assert len(join_cols) == 2

        join_tables = set()
        for col in join_cols:
            col.table = self.add_table(col.table, track=with_relations)
            join_tables.add(col.table)

        j = join or Join(
            operator='=',
            left=join_cols[0],
            right=join_cols[1]
        )

        if j not in self.joins:
            self._joins.add(j)
            if self.__track_order:
                self.__selections_order[j] = self.__next_index()

        self._joint_tables.add(cast(Tuple[TableRef, TableRef], tuple(join_tables)))

        return j

    def add_condition(self, cond: Condition) -> Condition:
        cond.col.table = self.add_table(cond.col.table)

        if cond not in self._conditions:
            self._conditions.add(cond)
            if self.__track_order:
                self.__selections_order[cond] = self.__next_index()

        return cond

    def add_selection(self, selection_col: ColumnRef, operator: str, operand: Any) -> Condition:
        return self.add_condition(Condition(
            col=selection_col,
            operator=operator,
            value=operand
        ))

    def add_projection(self, col: ColumnRef = None, func: str = None, projection: Projection = None) -> Projection:
        projection = projection or Projection(col, func)
        projection.col.table = self.add_table(projection.col.table)

        if projection not in self._projections:
            self._projections.add(projection)
            if self.__track_order:
                self.__projections_order[projection] = self.__next_index()

        return projection

    def are_joint(self, tables_to_join: List[TableRef]) -> bool:
        s = set()
        for t in tables_to_join:
            s.add(t)

        tup = cast(Tuple[TableRef, TableRef], tuple(s))
        if tup in self._joint_tables:
            return True
        return False

    def remove_condition(self, cond: Condition, cascade: bool = True) -> bool:
        self._conditions.remove(cond)

        if cascade and self.is_dangling(cond.col.table):
            self._relations.remove(cond.col.table)
            if self.__track_order:
                del self.__relations_order[cond.col.table]
            return True

        return False

    def remove_projection(self, projection: Projection, cascade: bool = True):
        self._projections.remove(projection)

        if cascade and self.is_dangling(projection.col.table):
            self._relations.remove(projection.col.table)
            if self.__track_order:
                del self.__relations_order[projection.col.table]
            return True

        # last projection clears the entire query
        # TODO: might want to change this behaviour to either not do anything,
        #  or replace with `select *`
        if not any(self._projections):
            self._relations = set()
            self._conditions = set()

            return True

        return False

    def remove_join(self, join: Join, cascade: bool = True) -> List[TableRef]:
        self._joins.remove(join)

        deleted = []
        for side in [join.left, join.right]:
            if cascade and self.is_dangling(side.table):
                self._relations.remove(side.table)
                if self.__track_order:
                    del self.__relations_order[side.table]
                deleted.append(side.table)

        return deleted

    def is_dangling(self, table: TableRef) -> bool:
        if any(p for p in self.projections if p.col.table == table):
            return False

        if any(c for c in self.conditions if c.col.table == table):
            return False

        if any(j for j in self.joins if j.left.table == table or j.right.table):
            return False

        return True

    def __copy__(self):
        q = Query()
        q._conditions = self._conditions.copy()
        q._projections = self._projections.copy()
        q._joins = self._joins.copy()
        q._relations = self._relations.copy()

        if self.__track_order:
            q.__track_order = True
            q.__selections_order = self.__selections_order.copy()
            q.__projections_order = self.__projections_order.copy()
            q.__relations_order = self.__relations_order.copy()

        q._joint_tables = self._joint_tables.copy()

        return q

    def copy(self):
        return self.__copy__()

    def to_sql(self, pretty=True, alias=True) -> str:
        # FIXME: this is mutating the existing query which is bad
        def selection_order(selection: Selection) -> str:
            if isinstance(selection, Condition):
                return '_' + selection.col.full_name
            if isinstance(selection, Join):
                return selection.left.full_name + '_' + selection.right.full_name

        # name tables
        if self.__track_order:
            relations = sorted(self.relations, key=lambda r: self.__relations_order[r])
            selections = sorted(self.selections, key=lambda s: self.__selections_order[s])
            projections = sorted(self.projections, key=lambda p: self.__projections_order[p])
        else:
            relations = sorted(self.relations, key=lambda r: r.name)
            selections = sorted(self.selections, key=selection_order)
            projections = sorted(self.projections, key=lambda p: p.col.name)

        if alias:
            for i, table in enumerate(relations):
                if not table.alias:
                    table.alias = f't{i + 1}'

        select_clause = ', '.join([f'{p}' for p in projections])
        from_clause = ', '.join([f'{t.name} as {t.alias}' if t.alias else t.name for t in relations])

        select = f"SELECT {select_clause} "
        frm = f"FROM {from_clause}"
        where = ""

        and_str = ' AND '
        if pretty:
            select += '\n'
            where += '\n'
            and_str += '\n' + ' ' * 6

        if any(selections):
            frm += ' '
            where += f"WHERE {and_str.join([str(s) for s in selections])}"

        q = select + frm

        if len(where) > 1:
            q += where

        return q

    def valid(self):
        if self.__len__() == 0:
            return False

        if len(list(self._projections)) == 0:
            return False

        # check projections and conditions
        for p in self._projections:
            if p.col.table not in self._relations:
                return False

        for c in self._conditions:
            if c.col.table not in self._relations:
                return False

        if len(self._relations) == 1 and len(self._joins) == 0:
            return True

        # check relations and joins
        orphan_relations = list(self._relations)
        orphan_joins = list(self._joins)

        while orphan_relations:
            relation = orphan_relations.pop()
            next_orphan_joins = [oj for oj in orphan_joins if oj.left.table != relation and oj.right.table != relation]
            if len(next_orphan_joins) == len(orphan_joins) and len(orphan_relations) > 0:
                # this means the is a relation that does not appear in any join
                return False
            orphan_joins = next_orphan_joins

        if len(orphan_joins) > 0:
            # this means there is a join without a from clause
            return False

        return True
