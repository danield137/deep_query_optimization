import os
import sys
import numpy as np
import pandas as pd
from pprint import pprint

if __name__ == '__main__':
    file_path = sys.argv[1]
    files = [file_path]
    if os.path.isdir(file_path):
        base_path = os.path.abspath(file_path)
        files = [
            os.path.join(base_path, f)
            for f in os.listdir(file_path) if (not os.path.isdir(f) and not f.startswith('.'))
        ]
    for file in files:
        df = pd.read_csv(file, names=['query', 'runtime'])
        df.logged = df.runtime.apply(np.log2).apply(lambda x: min(x, 8)).apply(lambda x: max(x, 0))
        pprint(f'{file}\n{df.logged.describe()}\n{"-" * 40}')
