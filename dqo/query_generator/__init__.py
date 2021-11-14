import random
import time
from typing import List, Set, cast, Optional

import numpy as np
import scipy.special
from tqdm import trange

from dqo.db.models import Database, Table, Column, DataType, ColumnStats, NumericStats
from dqo.random_utils import ChoicePreference, choose_with_preference, randint_with_preference
from dqo.relational import Query
from dqo.relational.models import Condition

query_types = ['simple', 'aggregate']
condition_types = ["<=", ">=", "=", "<>"]

cache = {}

OPERATORS = [">", "<", "=", "!="]

EPOCH_NOW = int(time.time())


def rand_condition(col: Column) -> Optional[Condition]:
    if col.data_type in (DataType.NUMBER, DataType.FLOAT):
        if not hasattr(col, 'stats') or not hasattr(col.stats, 'values') or not hasattr(col.stats.values, 'min') or not hasattr(col.stats, 'max'):
            rand_value = np.random.randint(0, 1e4)
        else:
            col_stats = cast(NumericStats, col.stats)
            rand_value = random.randint(col_stats.values.min, col_stats.values.max)

        rand_operator = random.choice(OPERATORS)
    elif col.data_type == DataType.TIME:
        if not hasattr(col, 'stats') or not hasattr(col.stats.values, 'min') or not hasattr(col.stats, 'max'):
            rand_value = np.random.randint(1, EPOCH_NOW)
        else:
            stats = cast(ColumnStats, col.stats)
            rand_value = random.randint(int(stats.values.min), int(stats.values.max))

        rand_value = f'to_timestamp(TRUNC(CAST({rand_value} AS bigint)))'

        rand_operator = random.choice(OPERATORS)
    elif col.data_type == DataType.STRING:
        rand_operator = 'LIKE'
        value = random.choices('abcdefghijklmnopqrstuvwxyz', k=3)
        rand_value = f"""'%{"".join(value)}%'"""

    elif col.data_type == DataType.BOOL:
        rand_operator = '='
        rand_value = f'{bool(random.getrandbits(1))}'
    else:
        raise ValueError(f'Unknown type "{col.data_type}" for {col.full_name}')

    return Condition(col.to_ref(), value=rand_value, operator=rand_operator)


class RandomQueryGen:
    def __init__(
            self,
            db: Database,
            seed: int = None
    ):
        self.db = db
        self.seed = seed
        self.query = Query()

    def choose_tables_to_use(self, population: List[Table]) -> List[Table]:
        k = choose_with_preference(list(range(1, len(population))), ChoicePreference.Left)

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        return random.choices(population, k=k)

    def choose_join_columns(self, join_tables: List[Table]) -> List[Column]:
        left, right = join_tables

        types_intersection = self.db[left.name].types_lookup.keys() & self.db[right.name].types_lookup.keys()

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        type_for_join = random.choice(list(types_intersection))

        left_col = random.choice(self.db[left.name].types_lookup[type_for_join])
        right_col = random.choice(self.db[right.name].types_lookup[type_for_join])

        return [left_col, right_col]

    def randomize_joins(self, limit, tables_subset: List[Table]):
        joined = []
        joinable = tables_subset[:]

        if self.seed:
            random.seed(self.seed)

        while len(joinable) > 0:
            if len(joined) > 0:
                left = random.choice(joined)
            else:
                left_index = random.choice(range(0, len(joinable)))
                left = joinable[left_index]
                joinable = joinable[:left_index] + joinable[left_index + 1:]
                joined.append(left)

            right_index = random.choice(range(0, len(joinable)))
            right = joinable[right_index]
            joinable = joinable[:right_index] + joinable[right_index + 1:]
            joined.append(right)

            join_columns = self.choose_join_columns([left, right])
            self.query.add_join([c.to_ref() for c in join_columns])

        # allow for extra variance
        possible_joins = int(scipy.special.comb(len(tables_subset), 2))
        max_join_count = min(limit, possible_joins)
        extra_join_count = min(possible_joins, max_join_count) - len(joined)
        chosen_extra_joins_count = random.choice(range(0, extra_join_count)) if extra_join_count > 0 else 0

        i = 0
        while i < chosen_extra_joins_count:
            join_tables = random.choices(tables_subset, k=2)
            if self.query.are_joint(join_tables):
                continue

            join_columns = self.choose_join_columns(join_tables)

            self.query.add_join([c.to_ref() for c in join_columns])
            i += 1

    def randomize_selections(self, limit, tables_subset: List[Table]):
        all_columns = [
            col
            for table in tables_subset
            for col in table.columns
        ]

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        num_selections = min(random.choice(range(len(all_columns))), limit)

        for rand_col in random.choices(all_columns, k=num_selections):
            condition = rand_condition(rand_col)
            if condition is not None:
                self.query.add_condition(condition)

    def randomize_projections(self, limit, tables_subset: List[Table]):
        for c in self.choose_columns(tables_subset, mx=limit):
            self.query.add_projection(c.to_ref(), 'MIN')

    def choose_columns(
            self,
            optional: List[Table] = None,
            must: List[Table] = None,
            mn: int = 1,
            mx: int = None
    ) -> Set[Column]:
        selected: Set[Column] = set()

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        must = must or []
        optional = optional or []
        _all = must + optional

        mn = max(mn, len(must)) if must else mn
        mx = mx or sum(
            1
            for t in _all
            for _ in t
        )

        all_columns: List[Set[Column]] = []
        # if there are must table, make sure we a grab a column of each
        if must:
            must_columns = [set(t.columns) for t in must]
            while len(selected) < mn:
                for m in must_columns:
                    c = random.choice(list(m))
                    m.remove(c)
                    selected.add(c)

                    if len(selected) == mx:
                        return selected

            all_columns = [t for t in must_columns if t]

        # if there is any point to continue
        if len(selected) < mx and (len(optional) > 0 or any(must)):
            if optional:
                for o in optional:
                    all_columns.append(set(o.columns))

            final_col_count = randint_with_preference(len(selected), mx, ChoicePreference.Left)

            while len(selected) < final_col_count and any(all_columns):
                for a in all_columns:
                    if a:
                        c = random.choice(sorted(list(a), key=lambda x: x.name))
                        a.remove(c)
                        selected.add(c)

                        if len(selected) == mx:
                            return selected

        return selected

    def randomize(self, max_joins=10, max_projections=20, max_predicates=30) -> Query:
        subset = self.choose_tables_to_use(self.db.tables)

        # TODO: currently, it's hard to handle multiple instances of the same table
        subset = list(set(subset))

        if len(subset) > 1:
            self.randomize_joins(max_joins, subset)

        self.randomize_selections(max_predicates, subset)
        self.randomize_projections(max_projections, subset)

        return self.query


def generate_queries(db: Database, n=1):
    queries = set()
    qg = RandomQueryGen(db)
    for _ in trange(0, n):
        queries.add(qg.randomize().to_sql(pretty=False).replace('\n', ' ') + '\n')

    return queries
