import json
import logging
import os
import sys
import time
from pathlib import Path

from dqo.datasets import QueriesDataset
from dqo.db.models import Database
from dqo.db.clients import Postgres
from dqo.query_generator.guided import BalancedQueryGen
from dqo.lab.query_executor import FileQueryExecutor, DatasetExecutor

logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

host = 'dqo-m.cayusbvr71xr.us-east-1.rds.amazonaws.com'
print(host)
print(sys.argv)

db_name = 'imdb' if len(sys.argv) == 1 else sys.argv[1]

if eval(os.environ.get('LOCAL_DB', 'False')):
    conn = f'https://localhost:5432/{db_name}'
    user = "postgres"
    password = "postgres"
else:
    conn = os.environ.get('DB_URI', f"https://{host}:5432/{db_name}")
    user = os.environ.get('DB_USER', "postgres")
    password = os.environ.get('DB_PASSWORD', "Dd150589!")

pg = Postgres(conn, user=user, password=password)

dataset_name = db_name
session_name = 'optimized'

schema_path = f'/Users/danieldubovski/projects/deep_query_optimization/dqo/datasets/{dataset_name}/execution/{session_name}/meta/schema.json'

if not os.path.exists(schema_path):
    schema = pg.model(use_cache=False)

    with open(schema_path, 'w+') as fp:
        json.dump(schema.as_dict(), fp)
    pg._schema = schema
else:
    pg._schema = Database.load(schema_path)

ds = QueriesDataset(f'{db_name}:optimized')
query_executor = DatasetExecutor(db_client=pg, ds=ds, extended=True, checkpoint=False)
print('executing...')
start_idx = 22791
query_executor.execute(start_idx=start_idx)
