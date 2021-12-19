import sys
import traceback

import pandas as pd

from dqo.estimator.neo.encoder import encode_query
from dqo.db.postgres import model_from_create_commands


def prepare_input(schema, query_log, output_path):
    db = model_from_create_commands(schema)

    import csv
    df = pd.DataFrame(columns=['input', 'runtime'])
    counter = 1
    with open(query_log) as f:
        queries = csv.reader(f)

        for query in queries:
            try:
                query_encoding = encode_query(db, query[0])
                df = df.append({'input': query_encoding, 'runtime': query[1]}, ignore_index=True)
            except Exception as ex:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f'FAILED PARSING QUERY (#{counter}): \n{type(ex).__name__} \n{query}')
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)

            counter += 1
        print(f'Parse query #{counter}')

    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    prepare_input(
        schema='../../data/job/data/schematext.sql',
        query_log='../../data/raw/query_log.csv',
        output_path='./input/encoded_queries.csv'
    )
