#!/usr/bin/env python

import logging

import click

from dqo.db.clients.postgres import Postgres
from dqo.query_generator import generate_queries

logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option('--host', '-h', required=True, help='Database host name (postgresql://dadubovs-postgres.postgres.database.azure.com:5432/imdb?sslmode=require)')
@click.option('--user', '-u', required=True, default='daduobvs@dadubovs-postgres')
@click.option('--password', '-p', required=True, default='Dd150589!')
@click.option('--count', '-c', default=100000, help='How many queries to execute')
@click.option('--batch', default=10000, help='How many queries to execute')
def execute(batch, count, password, user, host):
    logger.info(f'Starting. (host: {host}, count: {count}.')
    pg = Postgres(host, user=user, password=password)
    db = pg.model()

    queries = list(generate_queries(db, n=count))

    lines = len(queries)
    file_prefix = '{num}.txt'

    logger.info(f'dumping {len(queries)} queries to {lines // batch + 1} files: {file_prefix}')

    for i, b in enumerate(range(0, len(queries), batch)):
        current_file = file_prefix.format(num=i + 1)
        with open(current_file, 'w+') as f:
            f.writelines(queries[b:b + batch])
            logger.info(f'{current_file} done.')


if __name__ == '__main__':
    execute(password='postgres', user='postgres', host='http://localhost:5432/imdb')
