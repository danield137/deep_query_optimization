from dqo.db.clients import Postgres
import logging
import logging.config
import json
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    db_name = "tpch_alt"
    conn = f'https://dqo-m.cayusbvr71xr.us-east-1.rds.amazonaws.com/{db_name}'
    user = "postgres"
    password = "Dd150589!"
    pg = Postgres(conn, user=user, password=password)
    db = pg.model()
    print(f'db loaded ({len(db.tables)} tables found, with a total of {len(db.columns)} columns).')
