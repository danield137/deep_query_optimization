from typing import Union, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler

from dqo.estimator.greq.regressor_v2.dataset import BucketedQueriesDataset, ConcatQueriesDataSet


class QueriesDataModule(pl.LightningDataModule):
    def __init__(self, datasets: Union[str, List[str]]):
        super().__init__()
        self.dataset_names = [datasets] if type(datasets) is str else datasets

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.ds = ConcatQueriesDataSet([BucketedQueriesDataset(ds) for ds in self.dataset_names])

        df = self.ds.df.reset_index(drop=True)

        train_split, valid_split, test_split = 0.7, 0.2, 0.1
        min_rows = df.groupby('bucket').count().min()[0]

        sample_df = df.groupby('bucket').head(min_rows)
        others = df[~df.index.isin(sample_df.index)]
        others_len = len(others)

        train_df = sample_df.groupby('bucket').head(int(min_rows * train_split))
        train_df = train_df.append(others.sample(int(others_len * 0.3)))

        # print('train ', len(train_df))

        others = others[~others.index.isin(train_df.index)]
        val_df = sample_df[~sample_df.index.isin(train_df.index)].groupby('bucket').head(int(min_rows * valid_split))
        val_df = val_df.append(others.sample(len(val_df)))

        # print('val ', len(val_df))

        test_df = df[~df.index.isin(train_df.index)]
        test_df = test_df[~test_df.index.isin(val_df.index)]

        self.train_sampler = SubsetRandomSampler(train_df.index)
        self.valid_sampler = SubsetRandomSampler(val_df.index)
        self.test_sampler = SubsetRandomSampler(test_df.index)

    def train_dataloader(self):
        return DataLoader(self.ds, num_workers=0, batch_size=1, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.ds, num_workers=0, batch_size=1, sampler=self.valid_sampler)

    def test_dataloader(self):
        return DataLoader(self.ds, num_workers=0, batch_size=1, sampler=self.test_sampler)
