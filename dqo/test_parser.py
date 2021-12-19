import time
import numpy as np
from dqo.datasets import QueriesDataset
from dqo.relational import SQLParser
from dqo.relational.tree.parser import parse_ast
from dqo.relational.query.parser import parse_tree
from tqdm.auto import tqdm
from multiprocessing import Pool

if __name__ == '__main__':
    ds = QueriesDataset('imdb:small_uniform')
    df = ds.load()

    p = Pool(16)

    start = time.time()

    asts = [ast.root for ast in tqdm(p.imap(SQLParser.to_ast, df['query']), total=len(df))]
    took = time.time() - start
    print(took / len(df))
    assert len(df) == len(asts)

    start = time.time()
    trees = list(tqdm(p.imap(parse_ast, asts), total=len(asts)))
    took = time.time() - start
    print(took / len(df))

    assert len(df) == len(trees)

    start = time.time()
    queries = list(tqdm(p.imap(parse_tree, trees), total=len(trees)))
    took = time.time() - start
    print(took / len(df))

    assert len(df) == len(queries)