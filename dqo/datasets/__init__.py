import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dqo.db.models import Database
from dqo.relational import SQLParser
from dqo.relational.query.parser import parse_tree

try:
    from tqdm import tqdm
except:
    tqdm = None

logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def delete_folder(folder):
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class QueriesDataset:
    """
    Utility class to help load datasets
    """
    input_path: str
    meta_path: str
    df: pd.DataFrame
    loaded: Dict[str, int]
    columns: List[str] = ['query', 'runtime']
    column_types: List[type] = [str, float]
    chunk_size = 10 ** 5
    specific_file: Optional[str] = None
    converters = {"runtime": lambda x: float(x.replace("\"", ""))}

    def __init__(self, name: str, columns: Optional[List[str]] = None):
        datasets_path = os.path.dirname(__file__)
        dataset_name = name
        label = "latest"
        if ':' in dataset_name:
            parts = dataset_name.split(":")
            if len(parts) == 2:
                dataset_name, label = parts
            if len(parts) == 3:
                dataset_name, label, file_name = parts
                self.specific_file = file_name

        self.name = dataset_name
        self.label = label

        self.columns = columns or self.columns
        self.input_path = os.path.join(datasets_path, dataset_name, 'execution', label, 'runtimes')
        self.meta_path = os.path.join(datasets_path, dataset_name, 'execution', label, 'meta')
        self.df = pd.DataFrame(columns=self.columns)
        self.loaded = dict()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def list_files(self):
        if os.path.isfile(self.input_path):
            return [self.input_path]
        else:
            return sorted([
                os.path.join(self.input_path, f)
                for f in os.listdir(self.input_path) if '.csv' in f
            ])

    def schema(self) -> Database:
        return Database.load(os.path.join(self.meta_path, 'schema.json'))

    def load_splits(self, include_bucket=False):
        file_paths = self.list_files()
        train_df, test_df = None, None
        for idx, file_path in enumerate(file_paths):
            file_name, _ = os.path.splitext(os.path.basename(file_path))

            if file_name.lower() == 'test':
                test_df = pd.read_csv(
                    file_path, delimiter=",",
                    names=self.columns,
                    converters=self.converters

                )
            elif file_name.lower() == 'train':
                train_df = pd.read_csv(
                    file_path, delimiter=",",
                    names=self.columns,
                    converters=self.converters

                )

        if include_bucket:
            train_df['bucket'] = train_df.runtime.apply(np.log2).apply(np.round).apply(int).apply(lambda x: min(x, 8)).apply(lambda x: max(x, -3))
            test_df['bucket'] = test_df.runtime.apply(np.log2).apply(np.round).apply(int).apply(lambda x: min(x, 8)).apply(lambda x: max(x, -3))

        return train_df, test_df

    def load(self, include_bucket=False) -> pd.DataFrame:
        file_paths = self.list_files()
        # TODO: this can be done in parallel
        for idx, file_path in enumerate(file_paths):
            file_name, _ = os.path.splitext(os.path.basename(file_path))

            if self.specific_file is not None and file_name.lower() != self.specific_file.lower():
                continue

            if file_path not in self.loaded.keys():
                logger.info(f"Reading data from {file_path} [{idx + 1}/{len(file_paths)}]")
                _df = pd.read_csv(
                    file_path,
                    delimiter=",",
                    doublequote=True,
                    names=self.columns,
                    converters=self.converters
                )

                self.df = self.df.append(_df, ignore_index=True)
                self.loaded[file_path] = len(_df)
                logger.info(f"Loaded {len(_df)} rows from {file_path} [{idx + 1}/{len(file_paths)}]")
            else:
                logger.info(f'File {file_path} was already loaded, skipping [{idx + 1}/{len(file_paths)}]')
        self.df['bucket'] = self.df.runtime.apply(np.log2).apply(np.round).apply(int).apply(lambda x: min(x, 8)).apply(lambda x: max(x, -3))
        return self.df

    def append(self, file_path: str):
        pass

    def groom(self) -> pd.DataFrame:
        clean_df = pd.DataFrame(columns=self.columns)
        seen = set()

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            query = row['query'].strip()

            if query in seen:
                continue
            seen.add(query)

            try:
                rel_tree = SQLParser.to_relational_tree(query)
                rel_tree.optimize()
                row['nodes'] = len(rel_tree)
                row['parts'] = len(rel_tree.relations.keys()) + len(list(rel_tree.get_projections())) + len(list(rel_tree.get_selections()))

                # skip cartesian joins
                if len(rel_tree.root.children) > 1 or any([r for r in rel_tree.relations.values() if r.parent is None]):
                    continue
                # FIXME: this is due to a critical bug
                if row['nodes'] * 2 >= row['parts']:
                    clean_df = clean_df.append(row)
            except:
                pass

        return clean_df

    def groom_(self):
        self.df = self.groom()

    def augment(self, df: Optional[pd.DataFrame] = None):
        augmented_df = pd.DataFrame(columns=self.columns + ['nodes'])
        seen = set()
        df = df if df is not None else self.df
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            query = row['query'].strip()

            if query in seen:
                continue
            try:
                rel_tree = SQLParser.to_relational_tree(query)
                seen.add(query)
                copied = row.copy()
                copied['nodes'] = len(rel_tree)
                augmented_df = augmented_df.append(copied, ignore_index=True)

                permutations = rel_tree.permutations(limit=16)
                for permuted_tree in permutations:
                    permuted_q = parse_tree(permuted_tree, keep_order=True)
                    permuted_sql = permuted_q.to_sql(pretty=False, alias=False)
                    if permuted_sql not in seen:
                        seen.add(permuted_sql)
                        copied = row.copy()
                        copied['query'] = permuted_sql
                        copied['nodes'] = len(permuted_tree)
                        augmented_df = augmented_df.append(copied)

            except:
                pass
        return augmented_df

    def augment_(self):
        self.df = self.augment()

    def sample(self, strategy='random', n=None, frac=None):
        if len(self.df) == 0:
            if len(self.load()) == 0:
                return []

        df = self.df
        if strategy == 'random' or strategy == 'weighted':
            sample_kwargs = {}
            if strategy == 'weighted':
                df['bucket'] = df['runtime'].apply(np.log2).apply(np.round).apply(lambda x: max(x, 0)).apply(int)
                sample_kwargs.update(weights='bucket')
            if frac or n:
                if frac is not None:
                    sample_kwargs.update(frac=frac)
                elif n is not None:
                    sample_kwargs.update(n=min(n, len(df)))

                df = df.sample(**sample_kwargs)
        elif strategy == 'bucketed':
            df['bucket'] = df['runtime'].apply(np.log2).apply(np.round).apply(int)
            buckets = len(df['bucket'].value_counts())
            bucket_size = int(n / buckets) if n else 50

            df = df.groupby('bucket').head(bucket_size).copy()
        else:
            raise ValueError('Valid strategies are: random, bucketed, weighted')
        return df

    def save(self, prefix: str = 'data', replace=True, all_columns=False, schema: Database = None, split: bool = False):
        from pathlib import Path
        Path(self.input_path).mkdir(parents=True, exist_ok=True)
        if schema:
            Path(self.meta_path).mkdir(parents=True, exist_ok=True)

        if replace:
            delete_folder(self.input_path)

        if split:
            from sklearn.model_selection import train_test_split
            df = self.df.copy()
            df['bucket'] = df['runtime'].apply(np.log2).apply(np.round).apply(lambda x: max(x, -1)).apply(lambda x: min(x, 8)).apply(int)
            train_df, test_df = train_test_split(df, train_size=.9, stratify=df['bucket'])
            train_df.to_csv(os.path.join(self.input_path, 'train.csv'), header=False, index=False, columns=self.columns)
            test_df.to_csv(os.path.join(self.input_path, 'test.csv'), header=False, index=False, columns=self.columns)
        else:
            chunk_size = self.chunk_size
            for idx, start in enumerate(range(0, self.df.shape[0], chunk_size)):
                chunk = self.df.iloc[start:start + chunk_size]
                chunk_path = os.path.join(self.input_path, f'{prefix}_part_{idx:02d}{"" if replace else "_" + str(int(time.time()))}.csv')

                kwargs = dict(header=False, index=False)
                if not all_columns:
                    kwargs.update(columns=self.columns)
                chunk.to_csv(chunk_path, **kwargs)

        if schema:
            schema.save(os.path.join(self.meta_path, 'schema.json'))


class ExtendedQueriesDataset(QueriesDataset):
    columns = ['query', 'runtime', 'plan']
    column_types = [str, float, str]
    converters = {"runtime": lambda x: float(x.replace("\"", "")), "plan": lambda x: x.strip().replace('""', '"')}

    def augment_(self):
        raise NotImplementedError()

    def augment(self, df: Optional[pd.DataFrame] = None):
        raise NotImplementedError()
