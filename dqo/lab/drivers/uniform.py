import json
import logging
import os
import time
from pathlib import Path

from dqo.db.models import Database
from dqo.db.clients import Postgres
from dqo.query_generator.guided import BalancedQueryGen

logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

host = 'dqo-m.cayusbvr71xr.us-east-1.rds.amazonaws.com'
db_name = 'tpch'

if eval(os.environ.get('LOCAL_DB', 'False')):
    conn = f'https://localhost:5432/{db_name}'
    user = "postgres"
    password = "postgres"
else:
    conn = os.environ.get('DB_URI', f"https://{host}:5432/{db_name}")
    user = os.environ.get('DB_USER', "postgres")
    password = os.environ.get('DB_PASSWORD', "Dd150589!")

pg = Postgres(conn, user=user, password=password)

dataset_name = 'tpch'
session_alias = 'extended'

schema_path = f'/Users/danieldubovski/projects/deep_query_optimization/dqo/datasets/{dataset_name}/execution/{session_alias}/meta/schema.json'

if not os.path.exists(schema_path):
    schema = pg.model(use_cache=False)

    with open(schema_path, 'w+') as fp:
        json.dump(schema.as_dict(), fp)
    pg._schema = schema
else:
    pg._schema = Database.load(schema_path)

qg = BalancedQueryGen(pg, q_depth=50, checkpoint=True, extended=True)
qg.generate(100000)
