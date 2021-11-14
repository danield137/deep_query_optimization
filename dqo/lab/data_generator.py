from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import pandas as pd
import textdistance

from db.models import DataType, Column

fake_words = []

try:
    with open('/usr/share/dict/words', 'r') as words_file:
        fake_words = words_file.readlines()
except:
    ...


def common_prefix(strings) -> str:
    """ Find the longest string that is a prefix of all the strings.
    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix


def random_ints(min_val: int, hist: List[int], freq: List[int], total: int) -> List[int]:
    hist = [random_ints] + hist
    ranges: List[Tuple[int, int]] = []
    for i in range(len(hist) - 1):
        ranges.append((hist[i], hist[i + 1]))

    dist = [f / sum(freq) for f in freq]
    counts = [int(d * total) for d in dist]

    rands = []
    for i, rng in enumerate(ranges):
        rands += np.random.randint(low=rng[0], high=rng[1], size=counts[i])

    np.random.shuffle(rands)

    return rands


def random_floats(min_val: float, hist: List[float], freq: List[int], total: int) -> List[float]:
    hist = [random_ints] + hist
    ranges: List[Tuple[int, int]] = []
    for i in range(len(hist) - 1):
        ranges.append((hist[i], hist[i + 1]))

    dist = [f / sum(freq) for f in freq]
    counts = [int(d * total) for d in dist]

    rands = []
    for i, rng in enumerate(ranges):
        real = np.random.random(counts[i])
        natural = np.random.randint(low=int(rng[0]), high=int(rng[1]), size=counts[i])
        rands += np.add(real, natural)

    np.random.shuffle(rands)

    return rands


def random_strings(*args) -> List[str]:
    rands = []

    # hist = [random_ints] + hist
    # ranges: List[Tuple[int, int]] = []
    # for i in range(len(hist) - 1):
    #     ranges.append((hist[i], hist[i + 1]))
    #
    # dist = [f / sum(freq) for f in freq]
    # counts = [int(d * total) for d in dist]
    #
    #
    # for i, rng in enumerate(ranges):
    #     real = np.random.random(counts[i])
    #     natural = np.random.randint(low=int(rng[0]), high=int(rng[1]), size=counts[i])
    #     rands += np.add(real, natural)
    #
    # np.random.shuffle(rands)

    return rands


import abc


class StringGenerator(abc.ABC):
    @abc.abstractmethod
    def generate_values(self, n=1) -> List[str]:
        raise NotImplementedError()


@dataclass
class Pattern(StringGenerator):
    numeric: bool
    fmt: str
    lengths: List[int]
    prefix: Optional[str] = None

    def generate_values(self, n=1) -> List[str]:
        chars = 'abcdefghijklmnopqrstwxyzABCDEFGHIJKLMNOPQRSTWXYZ'
        nums = '0123456789'
        population = nums if self.numeric else chars

        r = []
        for i in range(n):
            parts = []
            for l in self.lengths:
                parts = []
                for i in range(n):
                    parts.append(''.join(np.random.choice(list(population), size=l)))
            gen = self.fmt.format(*parts)
            if self.prefix:
                gen = self.prefix + gen
            r.append(gen)
        return r


@dataclass
class Speech(StringGenerator):
    min_words: int
    max_words: int

    def generate_values(self, n=1) -> List[str]:
        r = []
        for i in range(n):
            length = np.random.randint(self.min_words, self.max_words)
            r.append(' '.join(np.random.choice(fake_words, size=length)))
        return r


@dataclass
class Categorical(StringGenerator):
    dist: Dict[str, int]

    def generate_values(self, n=1) -> List[str]:
        r = []
        total = np.sum(self.dist.values())
        for k, v in self.dist.items():
            n_proportional = int(n * v / total)
            r += [k] * n_proportional

        np.random.shuffle(r)
        return r


@dataclass
class Chars(StringGenerator):
    min_len: int
    max_len: int

    def generate_values(self, n=1) -> List[str]:
        r = []
        chars = 'abcdefghijklmnopqrstwxyzABCDEFGHIJKLMNOPQRSTWXYZ0123456789 '
        for i in range(n):
            length = np.random.randint(self.min_len, self.min_len)
            r.append(''.join(np.random.choice(list(chars), size=length)))
        return r


def infer_string_kind(series: pd.Series) -> StringGenerator:
    # pattern - customer#001 (similarity)
    # pattern - guid (hyphens)
    # categorical (distinct)
    # speech (words)
    # random chars (others)
    values = len(series)
    distinct = len(series.distinct())

    if distinct < values * 0.1:
        return Categorical(dict(series.value_counts()))

    spaces = []
    hyphens = []
    similarity = []
    prev = None
    lengths = []
    for text in series.items():
        if text:
            spaces.append(len(text.split(' ')))
            hyphens.append(len(text.split('-')))

            if prev is not None:
                similarity.append(textdistance.hamming.normalized_similarity(prev, text))
            prev = text
            lengths.append(len(text))

    if np.average(similarity) > 0.8:
        numeric = all(p.iddigit() for p in series[0].split('-'))
        prefix = common_prefix(series[:1000])
        fmt = '{0}'
        return Pattern(numeric=False, fmt=fmt, prefix=prefix, lengths=[np.max(lengths) - len(prefix)])
    if np.min(hyphens) == np.max(hyphens):
        # take one, find hyphens, count other part lengths, mark, check if numbers or strings
        part_lengths = [len(p) for p in series[0].split('-')]
        numeric = all(p.iddigit() for p in series[0].split('-'))

        fmt = '-'.join([f'{i}' for i, _ in enumerate(part_lengths)])
        return Pattern(numeric=numeric, fmt=fmt, lengths=part_lengths)
    if np.average(spaces) > 2:
        return Speech(np.min(spaces), np.max(spaces))

    return Chars(np.min(lengths), np.max(lengths))


def randomize_col(series: pd.Series, col: Column, n: int) -> List[Any]:
    if col.data_type in [DataType.NUMBER, DataType.TIME]:
        if series.is_monotonic:
            last = series.max()
            return list(range(last + 1, last + 1 + n))
        else:
            return random_ints(
                int(col.stats.values.min),
                [int(h) if h is not None else None for h in col.stats.values.hist],
                [int(f) if f is not None else None for f in col.stats.values.freq],
                n
            )

    elif col.data_type == DataType.FLOAT:
        return random_floats(
            col.stats.values.min, col.stats.values.hist, col.stats.values.freq, n
        )
    elif col.data_type == DataType.STRING:
        return infer_string_kind(series).generate_values(n)
    elif col.data_type == DataType.BOOL:
        raise NotImplementedError()

    raise ValueError(f'unexpected column type {col.data_type}')
