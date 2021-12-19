import sys
import os

if __name__ == '__main__':
    _, path = sys.argv
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    for file in files:
        with open(file+'.new', 'w+') as target:
            with open(file) as source:
                row = source.readline()
                while row:
                    cells = row.split(",")
                    target.write(",".join(f'"{c.strip()}"' for c in cells) + "\n")
                    row = source.readline()
