from typing import Tuple

import numpy as np
from scipy.special import comb

from dqo.relational import SQLParser
from dqo.relational.tree.node import RelationColumn


class Column:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        return f'Column({self.name}, {self.data_type})'


class Table:
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns

    def __repr__(self):
        return f'Table({self.name}, {self.columns})'


class Database:
    def __init__(self, tables):
        self.tables = tables


def index_as_int_array(value, padding) -> np.array:
    return np.fromstring(' '.join(bin(value)[2:].zfill(padding)), dtype=int, sep=' ')


operators_map = {
    val: i for i, val in
    enumerate(['<', '<=', '>', '>=', '!=', '=', 'NOT LIKE', 'BETWEEN', 'LIKE', 'IN', 'IS', 'IS NOT'])
}


def decompose_query(query: str, db: Database) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def column_key(table_name, column_name) -> str:
        return f'{table_name}::{column_name}'

    tables_count = len(db.tables)
    join_matrix = np.zeros((tables_count, tables_count), dtype=int)
    tables_map = {}
    columns_map = {}

    index = 0
    for t_col_index, t in enumerate(db.tables):
        tables_map[t.name] = t_col_index
        for c in t.columns:
            columns_map[column_key(t.name, c.name)] = index
            index += 1
    columns_count = index

    table_bits = int(np.ceil(np.log2(tables_count)))
    tables_sets = np.zeros((tables_count, table_bits), dtype=int)
    tables_mask = np.zeros(tables_count, dtype=float)
    # symmetric, so half is enough (upper right triangle of the matrix)
    joins_count = int(comb(tables_count, 2))
    join_bits = int(np.ceil(np.log2(joins_count)))
    join_sets = np.zeros((joins_count, join_bits), dtype=int)
    joins_mask = np.zeros(joins_count, dtype=int)
    # article suggests (col,op, normalized value), but in our case,
    # each predicate is represented by the column and operator (col,op)

    column_bits = int(np.ceil(np.log2(columns_count)))
    operator_bits = int(np.ceil(np.log2(len(operators_map.keys()))))
    predicates_sets = np.zeros((columns_count, column_bits + operator_bits), dtype=int)
    predicates_mask = np.zeros(columns_count, dtype=int)
    query_graph = SQLParser.to_relational_tree(query)

    for i, relation in enumerate(query_graph.relations.values()):
        tables_sets[i] = index_as_int_array(tables_map[relation.name], table_bits)
        tables_mask[i] = 1

    for join in query_graph.get_joins():
        l_index = tables_map[join.left.name]
        r_index = tables_map[join.right.name]

        join_matrix[l_index][r_index] = 1
        join_matrix[r_index][l_index] = 1

    joins = join_matrix[np.triu_indices(len(db.tables), k=1)]
    for i, join in enumerate(joins):
        if join == 1:
            join_sets[i] = index_as_int_array(i, join_bits)
            joins_mask[i] = 1

    for i, selection in enumerate(query_graph.get_selections(include_joins=False, flatten_or=True)):
        col = None
        op = None
        val = None

        if selection.operator in operators_map:
            op = operators_map[selection.operator]

        for operand in selection.operands:
            if isinstance(operand, RelationColumn):
                col = columns_map[column_key(operand.relation.name, operand.column)]
            else:
                val = operand.text

        predicate_encoded = [
            *index_as_int_array(col, column_bits),
            *index_as_int_array(op, operator_bits)
        ]
        predicates_sets[i] = predicate_encoded
        predicates_mask[i] = 1

    tables_sets = np.vstack(tables_sets)
    join_sets = np.vstack(join_sets)
    predicates_sets = np.vstack(predicates_sets)
    tables_mask = np.vstack(tables_mask)
    joins_mask = np.vstack(joins_mask)
    predicates_mask = np.vstack(predicates_mask)
    return tables_sets, join_sets, predicates_sets, tables_mask, joins_mask, predicates_mask


def encode_query(db: Database, query: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a query, and a database with T tables, J joins, C columns and O operators, encode the query
    :param db:
    :param query:
    :return:

    Table encoding - a matrix of shape (T,log(T)), where each row is the binary encoding of the table
    for example for 2 table out of 3 we will get [[0,1], [1,0],[0,0]]

    Join encoding - a matrix of shape (comb(T, 2),log(comb(T, 2)) , where each row is the existence of a specific join
    for example for 3 tables, there are 3 possible joins, given 1 of them we will get [[0,1], 0,0],[0,0]]

    Predicate encoding - a matrix of shape (C, log(C) + log(O)), such that each row represents the column and operator
    for example if we have 4 columns in the db, we need log(4) = 3 bits to represent it , and for 4 operators, we will
    need an extra 4 bits. so [0,0,1,0,   0,0,0,1] left part is the column, right part is the operator

    Masks - all of the masks are column matrices, representing which rows came from real values
    """
    clean_query = query.strip()
    if clean_query[-1] == ';':
        clean_query = clean_query[:-1]
    table_encoding, join_encoding, predicate_encoding, table_encoding_mask, join_encoding_mask, predicate_encoding_mask = decompose_query(clean_query, db)

    return table_encoding, join_encoding, predicate_encoding, table_encoding_mask, join_encoding_mask, predicate_encoding_mask
