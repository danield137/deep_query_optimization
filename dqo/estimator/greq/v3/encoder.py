import sys
from typing import List, cast

import numpy as np
import torch

from dqo.db.models import Database
from dqo.relational import RelationalTree
from dqo.relational import SQLParser
from dqo.relational.tree.node import RelationalNode, SelectionNode, RelationNode, \
    ProjectionNode, JoinNode, RelationColumn, OrNode


def index_as_int_array(value, padding) -> np.array:
    return np.fromstring(' '.join(bin(value)[2:].zfill(padding)), dtype=int, sep=' ')


operators_one_hot = {
    '<': [1, 0, 0],
    '<=': [1, 0, 0],
    '>': [1, 0, 0],
    '>=': [1, 0, 0],
    '!=': [0, 1, 0],
    '=': [0, 1, 0],
    'NOT LIKE': [0, 0, 1],
    'BETWEEN': [1, 0, 0],
    'LIKE': [0, 0, 1],
    'IN': [0, 1, 0],
    'IS': [0, 1, 0],
    'IS NOT': [0, 1, 0]
}

node_type_one_hot = {
    SelectionNode: [0, 0, 0, 1],
    # this isn't used - TODO: remove
    JoinNode: [0, 0, 1, 0],
    RelationNode: [0, 1, 0, 0],
    ProjectionNode: [1, 0, 0, 0],

}

types_one_hot = {
    'float': [1, 0, 0, 0, 0],
    'time': [0, 1, 0, 0, 0],
    'bool': [0, 0, 1, 0, 0],
    'number': [0, 0, 0, 1, 0],
    'string': [0, 0, 0, 0, 1],
}

types_index = {
    'float': 0,
    'time': 1,
    'bool': 2,
    'number': 3,
    'string': 4,
}

torch.set_default_dtype(torch.float64)


def encode_projection(projection: ProjectionNode, db: Database):
    node_encoding = torch.tensor(node_type_one_hot[ProjectionNode], dtype=torch.float64)
    types_encoding = torch.zeros(len(types_index.keys()) * 2)
    for c in projection.columns:
        db_column = db[c.relation.name][c.column]
        types_encoding[types_index[db_column.data_type] * 2] += 1
        types_encoding[types_index[db_column.data_type] * 2 + 1] += np.log10(db_column.stats.total)

    return torch.cat((node_encoding, types_encoding))


def encode_relation(relation: RelationNode, db: Database):
    node_encoding = torch.tensor(node_type_one_hot[RelationNode], dtype=torch.float64)
    types_encoding = torch.zeros(len(types_index.keys()) * 2)

    table = db[relation.name]
    for db_column in table.columns:
        types_encoding[types_index[db_column.data_type.value] * 2] += 1
        types_encoding[types_index[db_column.data_type.value] * 2 + 1] += np.log10(db_column.stats.total)

    return torch.cat((node_encoding, types_encoding))


COLUMN_ENCODING_LENGTH = 9


def encode_column(column: RelationColumn, db: Database) -> torch.Tensor:
    db_column = db[column.relation.name][column.column]
    return torch.tensor([
        np.log10(db_column.stats.total),
        db_column.stats.nulls_fraction,
        db_column.stats.distinct_ratio,
        *types_one_hot[db_column.data_type.value],
        int(db_column.stats.index)
    ], dtype=torch.float64)


def encode_selection(selection: SelectionNode, db: Database):
    node_encoding = torch.tensor(node_type_one_hot[SelectionNode], dtype=torch.float64)

    if isinstance(selection, OrNode):
        encoded = []
        for selection in cast(OrNode, selection).flatten_selections():
            encoded.append(encode_selection(selection, db))
        # TODO: should find a real way to encode this
        return torch.mean(torch.stack(encoded), 0)
    else:
        is_join = isinstance(selection, JoinNode)
        if is_join:
            left, right = selection.operands
            left_encoded = encode_column(left, db)
            right_encoded = encode_column(right, db)
            # equijoins only current
            op_encoded = torch.tensor(operators_one_hot[selection.operator.upper()], dtype=torch.float64)

            # 4 + 9 + 3 + 9 = 18 + 7
            return torch.cat((node_encoding, left_encoded, op_encoded, right_encoded))
        else:
            left, right = selection.operands
            relation_encoded = encode_column(left, db) if isinstance(left, RelationColumn) else encode_column(right, db)
            op_encoded = torch.tensor(operators_one_hot[selection.operator.upper()], dtype=torch.float64)
            # 4 + 9 + 3 = 
            return torch.cat((node_encoding, relation_encoded, op_encoded))


def encode_node(node: RelationalNode, db: Database):
    if isinstance(node, RelationNode):
        return encode_relation(node, db)
    if isinstance(node, ProjectionNode):
        return encode_projection(node, db)
    if isinstance(node, SelectionNode):
        return encode_selection(node, db)
    raise ValueError(f'Unexpected node type {type(node)}')


def encode_rel_tree(rel_tree: RelationalTree, db: Database) -> List[torch.Tensor]:
    encoded_nodes = []
    for node in rel_tree.dfs():
        encoded_node = encode_node(node, db)
        encoded_nodes.append(encoded_node)
    return encoded_nodes


def encode_query(db: Database, query: str) -> List[torch.Tensor]:
    clean_query = query.strip()
    if clean_query[-1] == ';':
        clean_query = clean_query[:-1]

    rel_tree = SQLParser.to_relational_tree(query)

    return encode_rel_tree(rel_tree, db)


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    from dqo.datasets import QueriesDataset
    from tqdm import tqdm

    ds = QueriesDataset(dataset_name)
    ds.load()
    db_schema = ds.schema()
    df = ds.df

    depths = []
    node_count = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        encode_query(db_schema, row['query'])
