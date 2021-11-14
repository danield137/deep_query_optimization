from typing import List, Callable, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dqo import datasets
from dqo.db.models import Database


# class TreeEncoder:
#     def __init__(self, db: Database):
#         self.db = db
#
#     def __call__(self, row):
#         row['input'] = encode_query(self.db, row['query'])
#
#         return row

class GereltEncode:
    def __init__(self, db: Database, encode_query: Callable[[Database, str], Any]):
        self.db = db
        self.encode_query = encode_query

    def __call__(self, row):
        row['input'] = self.encode_query(self.db, row['query'])

        return row


class BucketedQueriesDataset(Dataset):
    def __init__(self, db_name, encode_query: Callable[[Database, str], Any]):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ds = datasets.QueriesDataset(db_name)
        self.df = self.ds.load()
        self.transformers = [
            GereltEncode(self.ds.schema(), encode_query=encode_query)
        ]

        self.df['bucket'] = self.df.runtime.apply(np.log2).apply(np.round).apply(lambda x: min(x, 8)).apply(lambda x: max(x, -3)).apply(int)
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        row = self.df.iloc[idx]

        item = {
            'query': row['query'],
            'bucket': row['bucket'],
            'runtime': row['runtime']
        }
        for transform in self.transformers:
            item = transform(item)

        item = {
            'input': item['input'],
            'runtime': np.log2(item['runtime']),
            'bucket': item['bucket']
        }

        self.cache[idx] = item

        return item


class ConcatQueriesDataSet(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: List[BucketedQueriesDataset]):
        super().__init__(datasets)

        df = datasets[0].df

        for i in range(1, len(datasets)):
            df = df.append(datasets[i].df)

        self.df = df
