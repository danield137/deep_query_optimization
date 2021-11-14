from dataclasses import dataclass, field
from typing import Set

from dqo.datasets import QueriesDataset
from dqo.db.models import Database
from dqo.relational import Query, SQLParser


@dataclass
class DatasetAugmenter:
    dataset: QueriesDataset
    mem: Set[Query] = field(default_factory=set)

    def augment(self, factor=10):
        df = self.dataset.load()
        schema = self.dataset.schema()

        total_rows = len(df)
        percent = 0.2
        filtered = df.sort_values(by=['runtime'], ascending=False).head(int(total_rows * percent))

        for sql in filtered['queries']:
            query = SQLParser.to_query(sql)
