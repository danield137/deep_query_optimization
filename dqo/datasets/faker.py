import csv

import numpy as np
import pandas as pd

from dqo.datasets import QueriesDataset
from dqo.lab import data_generator


def generate(table, file_name, n):
    generated = []
    names = [c.name for c in table.columns]
    df = pd.read_csv(file_name, sep=',', names=names)

    generated = [[None] * n for _ in names]
    for i, col in enumerate(table.columns):
        real_n = n
        fake_nulls = 0
        gens = []

        if col.stats.nulls > 0:
            nulls_factor = col.stats.nulls / col.stats.rows
            fake_nulls = int(nulls_factor * n)
            real_n = n - fake_nulls

        gen = data_generator.randomize_col(df[col.name], col, real_n)

        if fake_nulls > 0:
            gens += [None] * fake_nulls
            np.shuffle(gens)

        generated[i] = gens

    rows = np.swapaxes(generated, 0, 1)

    # write file
    with open(f'gen_{file_name}', 'w+') as output:
        output.seek(0)
        writer = csv.writer(output, delimiter=',', quotechar='"')
        writer.writerows(generated)


if __name__ == '__main__':
    schema = QueriesDataset('tpch:optimized').schema()
    tasks = [
        (schema['region'], 'region.csv', 1000),
        (schema['nation'], 'nation.csv', 10000),
        (schema['supplier'], 'supplier.csv', 100000),
        (schema['part'], 'customer.csv', 300000),
        (schema['customer'], 'customer.csv', 1000000)

    ]
    for task in tasks:
        generate(*task)
