import time

from dqo.datasets import QueriesDataset
from dqo.estimator.gerelt import stats_based_encoder_pt as encoder

imdb = QueriesDataset("imdb:slow")

df = imdb.load()
db = imdb.schema()
counter = 0
took = []

for query in df['query'].values:
    start = time.process_time()
    encoded = encoder.encode_query(db, query)
    took.append(time.process_time() - start)

# 125 for each vector, max size 9250 => 74 nodes max tree size
print(f'for {len(took)} instances, took {sum(took)}, avg {sum(took) / len(took)}')
