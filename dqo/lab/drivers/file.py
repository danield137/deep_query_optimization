#!/usr/bin/env python

import click

from dqo.db.clients import Postgres
from dqo.lab.query_executor import FileQueryExecutor


@click.command()
@click.option('--shuffle', '-sh', default=True, help='Shuffle queries first')
@click.option('--count', '-c', default=-1, help='How many queries to execute')
@click.option('--host', '-h', required=True, help='postgresql://postgres@localhost:5432/imdb')
@click.option('--queries', default=None, help='path to query files')
@click.option('--existing', default=None, help='path to existing runs')
@click.option('--user', '-u', required=True)
@click.option('--password', '-p', required=True)
def execute(shuffle, count, host, queries, existing, user, password):
    """Simple program that greets NAME for a total of COUNT times."""

    print('connecting to: ' + host + ' with: ' + user)
    pg = Postgres(host, user=user, password=password)

    if queries is None:
        raise ValueError('--queries arg is mandatory in batch mode')
    query_executor = FileQueryExecutor(shuffle=shuffle, limit=count, db_client=pg, queries_path=queries, existing_path=existing)
    query_executor.execute()


if __name__ == '__main__':
    shuffle, count, host, queries, existing, user, password = (
        True, 10 ** 6,
        'http://localhost:5432/imdb',
        '/Users/danieldubovski/projects/deep_query_optimization/dqo/datasets/imdb/queries',
        '/Users/danieldubovski/projects/deep_query_optimization/dqo/runtimes/',
        'postgres',
        'postgres'
    )
    pg = Postgres(host, user=user, password=password)

    if queries is None:
        raise ValueError('--queries arg is mandatory in batch mode')
    query_executor = FileQueryExecutor(shuffle=shuffle, limit=count, db_client=pg, queries_path=queries, existing_path=existing)
    query_executor.execute()
