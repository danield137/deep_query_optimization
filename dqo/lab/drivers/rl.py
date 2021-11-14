import os
import time
from pathlib import Path

from dqo.db.models import Database
from dqo.db.postgres import Postgres
from dqo.query_generator.rl import EpisodicQueryGen

conn = "https://dqo-m.cocra5h3f2sc.us-east-1.rds.amazonaws.com:5432/imdb"
user = "postgres"
password = "DD150589!"
pg = Postgres(conn, user=user, password=password)

pg._schema = Database.load('../../datasets/imdb/execution/slow/meta/schema.json')
basedir = os.path.join('.', 'lab_results', str(int(time.time())))

Path(basedir).mkdir(parents=True, exist_ok=True)
ctx = {'basedir': basedir}

eqg = EpisodicQueryGen(pg, ctx=ctx)

counter = 0

eqg.run(episodes=100000)
