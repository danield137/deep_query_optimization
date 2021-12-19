from typing import Tuple

import numpy as np

from dqo.db.models import Database
from dqo.relational import SQLParser


def decompose_query(query: str, db) -> Tuple[np.array, np.array]:
    def column_key(table_name, column_name) -> str:
        return f'{table_name}::{column_name}'

    n = len(db.tables)
    join_matrix = np.zeros((n, n), dtype=int)
    tables_map = {}
    columns_map = {}

    index = 0
    for t_col_index, t in enumerate(db.tables):
        tables_map[t.name] = t_col_index
        for c in t.columns:
            columns_map[column_key(t.name, c.name)] = index
            index += 1

    predicates_array = np.zeros(index, dtype=int)
    query_graph = SQLParser.to_relational_tree(query)

    for join in query_graph.get_joins():
        l_index = tables_map[join.left.name]
        r_index = tables_map[join.right.name]

        join_matrix[l_index][r_index] = 1
        join_matrix[r_index][l_index] = 1

    for predicate_column in query_graph.get_selection_columns():
        col_index = columns_map[column_key(predicate_column.relation.name, predicate_column.column)]
        predicates_array[col_index] = 1

    return join_matrix, predicates_array


def encode_query(db: Database, query: str):
    clean_query = query.strip()
    if clean_query[-1] == ';':
        clean_query = clean_query[:-1]
    join_encoding, predicate_encoding = decompose_query(clean_query, db)

    return [*join_encoding[np.triu_indices(len(db.tables), k=1)], *predicate_encoding]
