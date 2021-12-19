import logging
from typing import Union, List, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler

from dqo.estimator.gerelt.dataset import BucketedQueriesDataset, ConcatQueriesDataSet

SplitFunction = Callable[[pd.DataFrame, List[float]], List[Sampler]]
logger = logging.getLogger('dqo.estimator.data_module')


def smart_split(df: pd.DataFrame, splits: List[float] = (.6, .2, .2)) -> List[Sampler]:
    min_rows = df.groupby('bucket').count().min()[0]
    sample_df = df.groupby('bucket').head(min_rows)
    others = df[~df.index.isin(sample_df.index)]
    others_len = len(others)

    if len(splits) == 3:
        train_split, valid_split, test_split = splits

        train_df = sample_df.groupby('bucket').head(int(min_rows * train_split))
        train_df = train_df.append(others.sample(int(others_len * 0.3)))

        # print('train ', len(train_df))

        others = others[~others.index.isin(train_df.index)]
        val_df = sample_df[~sample_df.index.isin(train_df.index)].groupby('bucket').head(int(min_rows * valid_split))
        val_df = val_df.append(others.sample(len(val_df)))

        test_df = df[~df.index.isin(train_df.index)]
        test_df = test_df[~test_df.index.isin(val_df.index)]

        return [SubsetRandomSampler(train_df.index), SubsetRandomSampler(val_df.index), SubsetRandomSampler(test_df.index)]
    elif len(splits) == 2:
        train_split, valid_split = splits

        train_df = sample_df.groupby('bucket').head(int(min_rows * train_split))
        train_df = train_df.append(others.sample(int(others_len * 0.8)))

        others = others[~others.index.isin(train_df.index)]
        val_df = sample_df[~sample_df.index.isin(train_df.index)]
        val_df = val_df.append(others.sample(len(val_df)))

        return [SubsetRandomSampler(train_df.index), SubsetRandomSampler(val_df.index)]


def uniform_split(df: pd.DataFrame, splits: List[float] = (.6, .2, .2)) -> List[Sampler]:
    mean_rows = int(df.groupby('bucket').count().mean()[0])
    sample_df = df.groupby('bucket').head(mean_rows)

    return default_split(sample_df, splits)


def default_split(df: pd.DataFrame, splits: List[float] = (.6, .2, .2)) -> List[Sampler]:
    if len(splits) == 2:
        train_split, valid_split = splits
        train_df, val_df = train_test_split(df, train_size=train_split / (train_split + valid_split), shuffle=True, stratify=df['bucket'])

        return [SubsetRandomSampler(train_df.index), SubsetRandomSampler(val_df.index)]
    elif len(splits) == 3:
        train_split, valid_split, test_split = splits

        train_df, test_df = train_test_split(df, בtrain_size=train_split + valid_split, shuffle=True, stratify=df['bucket'])
        train_df, val_df = train_test_split(train_df, train_size=train_split / (train_split + valid_split), shuffle=True, stratify=train_df['bucket'])

        return [SubsetRandomSampler(train_df.index), SubsetRandomSampler(val_df.index), SubsetRandomSampler(test_df.index)]


class QueriesDataModule(pl.LightningDataModule):
    train_sampler: Sampler
    valid_sampler: Sampler
    test_sampler: Sampler
    split_fn: SplitFunction

    def __init__(self, datasets: Union[str, List[str]], encoder, split_kind: str = 'smart', frac: float = 1.0, split=(.8, .2), value_transform=np.log2):
        """
        :param datasets:
        :param split_kind: valid options are ('smart', 'uniform', 'default' or None)
        """
        super().__init__()

        if split_kind == 'smart':
            self.split_fn = smart_split
        elif split_kind == 'uniform':
            self.split_fn = uniform_split
        else:
            self.split_fn = default_split

        self.frac = frac
        self.encoder = encoder
        self.value_transform = value_transform
        self.split = split
        self.dataset_names = [datasets] if type(datasets) is str else datasets

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.ds = ConcatQueriesDataSet(
            [BucketedQueriesDataset(ds, encode_query=self.encoder.encode_query, value_transform=self.value_transform) for ds in self.dataset_names])
        df = self.ds.df.reset_index(drop=True)

        if self.frac < 1:
            original_size = len(df)
            df = df.groupby('bucket').apply(lambda x: x.iloc[:max(int(x['bucket'].size * self.frac), 3)]).reset_index(drop=True)
            logger.info(f'taking a fraction {self.frac} of rows ({len(df)}/{original_size})')

        samplers = self.split_fn(df, splits=self.split)
        if len(samplers) == 2:
            self.train_sampler, self.valid_sampler = samplers
            logger.info(f'dataset sizes (train, val): {len(self.train_sampler)},{len(self.valid_sampler)}')
        elif len(samplers) == 3:
            self.train_sampler, self.valid_sampler, self.test_sampler = samplers
            logger.info(f'dataset sizes (train, val, test): {len(self.train_sampler)},{len(self.valid_sampler)},{len(self.test_sampler)}')

    def train_dataloader(self):
        return DataLoader(self.ds, num_workers=0, batch_size=1, sampler=self.train_sampler, collate_fn=lambda x: x)

    def val_dataloader(self):
        return DataLoader(self.ds, num_workers=0, batch_size=1, sampler=self.valid_sampler, collate_fn=lambda x: x)

    def test_dataloader(self):
        if self.test_sampler is None:
            return None

        return DataLoader(self.ds, num_workers=0, batch_size=1, sampler=self.test_sampler)
