from dqo.datasets import QueriesDataset

ds = QueriesDataset('imdb:resp_time_clean')
ds.load()
ds.groom()