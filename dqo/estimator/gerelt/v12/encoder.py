import enum
import sys
from collections import defaultdict
from typing import Tuple, Any, List

import numpy as np
import pendulum
import scipy.stats
import torch

from dqo.db.models import Database, DataType
from dqo.relational import RelationalTree
from dqo.relational import SQLParser
from dqo.relational.tree.node import RelationalNode, SelectionNode, RelationNode, \
    ProjectionNode, JoinNode, RelationColumn, OrNode
from dqo.tree import Node


class EncodedNode(Node):
    ...


class EncodedSelection(EncodedNode):
    ...


class EncodedJoin(EncodedNode):
    ...


class EncodedProjection(EncodedNode):
    ...


class EncodedRelation(EncodedNode):
    ...


def summarize_set(s):
    """
    returns a vector of length 8
    :param s:
    :return:
    """
    stats = scipy.stats.describe(s)

    return [
        len(s),
        stats.minmax[0],
        stats.minmax[1],
        stats.mean,
        np.median(s),
        np.nan_to_num(stats.variance),
        np.nan_to_num(stats.skewness),
        np.nan_to_num(stats.kurtosis),
    ]

def index_as_int_array(value, padding) -> np.array:
    return np.fromstring(' '.join(bin(value)[2:].zfill(padding)), dtype=int, sep=' ')


# 10 buckets, 1 for nulls
HIST_MAX_LEN = 11

operator_kind_onehot = {
    'eq': [0, 0, 0, 0, 1],
    'neq': [0, 0, 0, 1, 0],
    'range': [0, 0, 1, 0, 0],
    'sim': [0, 1, 0, 0, 0],
    'nsim': [1, 0, 0, 0, 0]
}

operators_one_hot = {
    '<': operator_kind_onehot['range'],
    '<=': operator_kind_onehot['range'],
    '>': operator_kind_onehot['range'],
    '>=': operator_kind_onehot['range'],
    '!=': operator_kind_onehot['neq'],
    '=': operator_kind_onehot['range'],
    'NOT LIKE': operator_kind_onehot['nsim'],
    'BETWEEN': operator_kind_onehot['range'],
    'LIKE': operator_kind_onehot['sim'],
    'IN': operator_kind_onehot['eq'],
    'IS': operator_kind_onehot['eq'],
    'IS NOT': operator_kind_onehot['neq']
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

english_letter_probability = {
    'e': 0.1202,
    't': 0.910,
    'a': 0.812,
    'o': 0.768,
    'i': 0.731,
    'n': 0.695,
    's': 0.628,
    'r': 0.602,
    'h': 0.592,
    'd': 0.432,
    'l': 0.398,
    'u': 0.288,
    'c': 0.271,
    'm': 0.261,
    'f': 0.230,
    'y': 0.211,
    'w': 0.209,
    'g': 0.203,
    'p': 0.182,
    'b': 0.149,
    'v': 0.111,
    'k': 0.069,
    'x': 0.017,
    'q': 0.011,
    'j': 0.010,
    'z': 0.007,
}

torch.set_default_dtype(torch.float64)


def estimate_size(d_type: DataType) -> float:
    if d_type == DataType.NUMBER:
        return 0.2
    if d_type == DataType.BOOL:
        return 0.01
    if d_type == DataType.STRING:
        return 1
    if d_type == DataType.FLOAT:
        return 0.3
    if d_type == DataType.TIME:
        return 0.2


def adjusted_value_and_probability(hist: List[float], freq: List[int], value: float, agg: bool = False) -> Tuple[float, float]:
    bucket = 0

    if not hist:
        return 0, 0

    if hist[-1] is None:
        if value is None:
            return 1, freq[-1] / sum(freq)
        else:
            hist = hist[:-1]
            freq = freq[:-1]

    num_buckets = len(hist)

    for bucket_idx, ceil in enumerate(hist):
        if value < ceil:
            break
        bucket = bucket_idx

    total_items = sum(freq)

    if agg:
        adjusted_value = sum(range(bucket, num_buckets)) / (num_buckets - bucket)
        value_probability = sum([
            freq[i] for i in range(bucket, num_buckets)
        ]) / total_items
    else:
        adjusted_value = (bucket + 1) / num_buckets
        value_probability = freq[bucket] / total_items

    return adjusted_value, value_probability


def encode_col_hist_freq(
        col: RelationColumn, db: Database, fixed_length=HIST_MAX_LEN,
        pad_values: Tuple[float, float] = (1e-8, .1)
) -> Tuple[List[float], List[float]]:
    db_col = db[col.relation.name][col.column]

    if db_col.data_type == DataType.STRING:
        hist = db_col.stats.values.length.hist
        freq = db_col.stats.values.length.freq
    else:
        hist = db_col.stats.values.hist
        freq = db_col.stats.values.freq

    if len(hist) != len(freq):
        raise ValueError()

    if len(hist) > 11 or len(freq) > 11:
        raise NotImplementedError()

    hist_pad, freq_pad = pad_values
    padding = fixed_length - len(hist)
    hist = list(np.log([np.abs(v) if v else hist_pad for v in hist]))
    freq = list(np.log10([v if v else freq_pad for v in freq]))
    if padding:
        hist += ([np.log(hist_pad)] * padding)
        freq += ([np.log10(freq_pad)] * (fixed_length - len(freq)))

    return hist, freq


def encode_col_measures(db: Database, rel_col: RelationColumn = None, col: str = None, table: str = None, zero_fill=1e-8) -> Tuple[float, float, float]:
    if rel_col:
        col = rel_col.column
        table = rel_col.relation.name

    db_col = db[table][col]

    if db_col.data_type == DataType.STRING:
        variance = db_col.stats.values.length.variance
        skewness = db_col.stats.values.length.skewness
        kurtosis = db_col.stats.values.length.kurtosis
    else:
        variance = db_col.stats.values.variance
        skewness = db_col.stats.values.skewness
        kurtosis = db_col.stats.values.kurtosis

    variance = np.log(np.abs(variance)) if variance and not np.isnan(variance) else 0
    skewness = np.log(np.abs(skewness)) if skewness and not np.isnan(skewness) else 0
    kurtosis = np.log(np.abs(kurtosis)) if kurtosis and not np.isnan(kurtosis) else 0

    return variance, skewness, kurtosis


def encode_column(db: Database, table: str = None, column: str = None, rel_col: RelationColumn = None) -> torch.Tensor:
    """
    A column is encoded as follows:
    ------------------------------------------------------------------------------------------------------
    | # items | % distinct | % nulls | size | indexed | variance | skewness | kurtosis | onehot datatype |
    ------------------------------------------------------------------------------------------------------

    Vector size is 8 + 5 = 13
    """
    if rel_col:
        table = rel_col.relation.name
        column = rel_col.column

    db_column = db[table][column]

    return torch.tensor([
        np.log10(db_column.stats.total),
        db_column.stats.distinct_ratio,
        db_column.stats.nulls_fraction,
        estimate_size(db_column.data_type),
        int(db_column.stats.index),
        *encode_col_measures(db, rel_col=rel_col),
        *types_one_hot[db_column.data_type.value],
    ], dtype=torch.float64)


def encode_operand(
        db: Database, operand: Any, table: str = None, column: str = None, rel_col: RelationColumn = None
) -> List[float]:
    if rel_col:
        table = rel_col.relation.name
        column = rel_col.column

    db_column = db[table][column]

    # TODO: encode value using schema meta data like a historgram
    if db_column.data_type == DataType.STRING:
        value = operand.text.replace('%', '').replace("'", '')
        text_len = len(value) * 1.0

        left_wildcard = operand.text[1] == '%'
        right_wildcard = operand.text[-2] == '%'

        # TODO: A better approach would probably be to check the frequency of each letter, and use that instead..
        letter_probabilities = 1
        for letter in value.lower():
            letter_probabilities *= english_letter_probability[letter]

        adjusted_value, value_probability = adjusted_value_and_probability(
            db_column.stats.values.length.hist,
            db_column.stats.values.length.freq,
            text_len,
            agg=left_wildcard or right_wildcard
        )

        return [
            int(left_wildcard),
            adjusted_value,
            value_probability * letter_probabilities,
            int(right_wildcard)
        ]
    if db_column.data_type == DataType.BOOL:
        value = 1 if operand.text.lower() in ['true', '1'] else 0
        return [
            0,
            *adjusted_value_and_probability(db_column.stats.values.hist, db_column.stats.values.freq, value),
            0
        ]
    if db_column.data_type == DataType.TIME:
        value = pendulum.parse(operand.text).float_timestamp
        adjusted, prob = adjusted_value_and_probability(db_column.stats.values.hist, db_column.stats.values.freq, value)
        return [
            0,
            adjusted - 0.5,
            prob,
            0
        ]
    if db_column.data_type in [DataType.NUMBER, DataType.FLOAT]:
        value = float(operand.text.lower())
        adjusted, prob = adjusted_value_and_probability(db_column.stats.values.hist, db_column.stats.values.freq, value)
        return [
            0,
            # center around the middle
            adjusted - 0.5,
            prob,
            0
        ]


def encode_projection(projection: ProjectionNode, db: Database) -> EncodedProjection:
    """
    Given P projections on C columns, we encode it as follows:
    1. Stats (denoted S) are encoded using a matrix of size C x S:
     ___________________________________________________________________________
    |    values  | distinct % | nulls % | size | variance | skewness | kurtosis |
    |---------------------------------------------------------------------------|
    |     ...    |   ...      |   ...   |  ... |   ...   |   ...   |      ...   |
    | ------------------------------------------------------------------------- |
    |     ...    |   ...      |   ...   |  ... |   ...   |   ...   |      ...   |
    |___________________________________________________________________________|

    These are then reduce using min/max/mean/variance/skw/kurtis (scipy.stats.describe) to
    conform to a 1d vector of size S.
    That vector is later concatenated with table wide metrics:
    ---------------------------------------------------------------------------------------
    | stats | # int columns | # string columns | # float cols | # bool cols | # date cols |
    ---------------------------------------------------------------------------------------
    Vector size is: |S| * |Rs| + 5 = 47
    """

    column_stats = []
    column_types = defaultdict(int)

    for rel_col in projection.columns:
        table = rel_col.relation
        col = rel_col.column

        db_col = db[table.name][col]
        column_types[db_col.data_type.lower()] += 1

        column_stats.append([
            np.log10(db_col.stats.total),
            db_col.stats.distinct_ratio,
            db_col.stats.nulls_fraction,
            estimate_size(db_col.data_type),
            int(db_col.stats.index),
            *encode_col_measures(db, rel_col=rel_col)
        ])

    column_stats = np.array(column_stats)

    cum_column_stats = []

    for i in range(column_stats.shape[1]):
        cum_column_stats.append(summarize_set(column_stats[:, i]))

    type_counts = [
        np.log(column_types['string']) if 'string' in column_types else 0,
        np.log(column_types['number']) if 'number' in column_types else 0,
        np.log(column_types['float']) if 'float' in column_types else 0,
        np.log(column_types['bool']) if 'bool' in column_types else 0,
        np.log(column_types['time']) if 'time' in column_types else 0
    ]

    return EncodedProjection(
        value=torch.Tensor([
            *np.array(cum_column_stats).reshape(-1),
            *type_counts
        ])
    )


def encode_relation(relation: RelationNode, db: Database) -> EncodedRelation:
    """
    Given a table T with C columns, we encode it as follows:
    1. Stats (denoted S) are encoded using a matrix of size C x S:
    _________________________________________________________________________
    | distinct % | nulls % | size | indexed | variance | skewness| kurtosis |
    |-----------------------------------------------------------------------|
    |     ...    |   ...   | ...  |    ...  |    ...   |   ...   |   ...    |
    | --------------------------------------------------------------------- |
    |     ...    |   ...   | ...  |    ...  |    ...   |   ...   |   ...    |
    |_______________________________________________________________________|

    These are then reduce using min/max/mean/variance/skw/kurtis (scipy.stats.describe) to
    conform to a 1d vector of size S.
    That vector is later concatenated with table wide metrics:
    -------------------------------------------------------------------------------------------------------
    | stats | # int columns | # string columns | # float cols | # bool cols | # date cols | # rows | size |
    -------------------------------------------------------------------------------------------------------

    Which gives us a vector of size: |S * 6 channels| + 5 + 1 + 1 , in the above case, where |S| is 7, we get a vector of size (49).
    :param relation:
    :param db:
    :return:
    """
    table = db[relation.name]
    column_stats = []
    column_types = defaultdict(int)

    for column in table.columns:
        column_types[column.data_type] += 1
        column_stats.append([
            column.stats.distinct_ratio,
            column.stats.nulls_fraction,
            estimate_size(column.data_type),
            int(column.stats.index),
            *encode_col_measures(db, table=table.name, col=column.name)
        ])

    column_stats = np.array(column_stats)

    cum_column_stats = []
    for i in range(column_stats.shape[1]):
        cum_column_stats.append(summarize_set(column_stats[:, i]))

    return EncodedRelation(
        value=torch.Tensor([
            *np.array(cum_column_stats).reshape(-1),
            np.log(column_types['string']) if 'string' in column_types else 0,
            np.log(column_types['number']) if 'number' in column_types else 0,
            np.log(column_types['float']) if 'float' in column_types else 0,
            np.log(column_types['bool']) if 'bool' in column_types else 0,
            np.log(column_types['time']) if 'time' in column_types else 0,
            np.log10(table.stats.rows),
            np.log10(table.stats.pages * table.stats.page_size)
        ])
    )


def encode_join(join: JoinNode, db: Database) -> EncodedJoin:
    """
    Encoding is a vector of:

     --------------------------------------------------------------------------------------------------------------------
     | *l_col | *l_rel | *l_hist | *l_freq | l_indexed | *r_col | *r_rel | *r_hist | *l_freq | r_indexed | onehot dtype |
     --------------------------------------------------------------------------------------------------------------------

    Vector size is then: 2 (|C| + |R| + |H| + |F| + 1) + 5. given |C| = 13 , |R| = 49, |H| = 11 |F| = 11, we get |V| = 175
    """
    left, right = join.operands

    join_vector = []
    for side in [left, right]:
        col = encode_column(db=db, rel_col=side).numpy()
        rel = encode_relation(side.relation, db).value.numpy()
        hist, freq = encode_col_hist_freq(side, db)
        db_col = db[side.relation.name][side.column]

        join_vector = [*join_vector, *col, *rel, *hist, *freq, int(db_col.stats.index)]

    return EncodedJoin(torch.Tensor(join_vector + types_one_hot[db_col.data_type]))


def encode_selection(selection: SelectionNode, db: Database) -> EncodedSelection:
    """
       Encoding is a vector of:

        -------------------------------------------------------------------------------
        | *col | *rel |  *hist | *freq | onehot dtype | op one hot | operand encoding |
        -------------------------------------------------------------------------------

       Vector size is then: |C| + |R| + |H| + |F| + 5 + 5 + 4. given |C| = 13, |R| = 49, |H| = 11, |F| = 10, we get |V| = 98

       """
    if isinstance(selection, OrNode):
        raise NotImplementedError()
        # encoded = []
        # for selection in cast(OrNode, selection).flatten_selections():
        #     encoded.append(encode_selection(selection, db).value)
        # return EncodedSelection(torch.mean(torch.stack(encoded), 0))

    left, right = selection.operands

    if isinstance(left, RelationColumn):
        rel_col, operand = left, right
    else:
        rel_col, operand = right, left

    col = encode_column(db=db, rel_col=rel_col).numpy()
    rel = encode_relation(db=db, relation=rel_col.relation).value.numpy()
    hist, freq = encode_col_hist_freq(rel_col, db)
    db_col = db[rel_col.relation.name][rel_col.column]

    encoded_operand = encode_operand(rel_col=rel_col, operand=operand, db=db)
    op = operators_one_hot[selection.operator.upper()]

    return EncodedSelection(
        torch.Tensor([
            *col,
            *rel,
            *hist,
            *freq,
            *types_one_hot[db_col.data_type],
            *operators_one_hot[selection.operator],
            *encoded_operand
        ])
    )


class NodeType(enum.Enum):
    JOIN = 'join',
    SELECTION = 'selection'
    PROJECTION = 'projection'
    OR = 'or'
    RELATION = 'relation'


def encode_node(node: RelationalNode, db: Database) -> Node[Tuple[NodeType, torch.Tensor]]:
    encoded = None

    if isinstance(node, RelationNode):
        encoded = encode_relation(node, db)
    elif isinstance(node, ProjectionNode):
        encoded = encode_projection(node, db)
    elif isinstance(node, JoinNode):
        encoded = encode_join(node, db)
    elif isinstance(node, SelectionNode):
        encoded = encode_selection(node, db)
    else:
        raise ValueError(f'Unexpected node type {type(node)}')

    return encoded


def encode_rel_tree(rel_tree: RelationalTree, db: Database) -> List[torch.Tensor]:
    encoded_nodes = []

    rel_tree.optimize()
    for node in rel_tree.dfs():
        encoded_nodes.append(encode_node(node, db).value)

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
