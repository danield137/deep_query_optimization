import json
import sys
import traceback

import pandas as pd

from dqo.db.postgres import model_from_create_commands
from dqo.estimator.kipf.encoder import encode_query


def prepare_input(schema, query_log, output_path):
    db = model_from_create_commands(schema)

    import csv
    df = pd.DataFrame(columns=['input', 'runtime'])
    counter = 1
    with open(query_log) as f:
        queries = csv.reader(f)

        for query in queries:
            try:
                tables, joins, predicates, tables_mask, joins_mask, predicates_mask = encode_query(db, query[0])
                df = df.append({
                    'input_tables': json.dumps(tables.tolist()),
                    'input_tables_mask': json.dumps(tables_mask.tolist()),
                    'input_joins': json.dumps(joins.tolist()),
                    'input_joins_mask': json.dumps(joins_mask.tolist()),
                    'input_predicates': json.dumps(predicates.tolist()),
                    'input_predicates_mask': json.dumps(predicates_mask.tolist()),
                    'runtime': query[1]
                }, ignore_index=True)
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
