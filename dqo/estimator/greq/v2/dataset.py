from typing import Union, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from dqo import datasets
from dqo.db.models import Database
from dqo.estimator.greq.regressor_v2.gerelt_encoder import encode_query


class GereltEncode:
    def __init__(self, db: Database):
        self.db = db

    def __call__(self, row):
        row['input'] = encode_query(self.db, row['query'])

        return row


class BucketByLog2Runtime:
    def __init__(self, bounds=(1, 2 ** 8 - 1)):
        self.min_bound, self.max_bound = bounds

    def __call__(self, row):
        runtime = row['runtime']
        row['bucket'] = np.round(np.max(np.min(np.log2(runtime), 0), 8))

        return row


class BucketedQueriesDataset(Dataset):
    def __init__(self, db_name):
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
            GereltEncode(self.ds.schema())
        ]

        self.df['bucket'] = self.df.runtime.apply(np.log2).apply(lambda x: min(x, 8)).apply(lambda x: max(0, x)).apply(np.round).apply(int)
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