from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Generic

import numpy as np

T = TypeVar('T')


@dataclass
class Partitioner(Generic[T], ABC):
    k: int

    def partition(self, v: T) -> int:
        """
        Given a value v, return the partition
        :param v:
        :return:
        """
        raise NotImplementedError()

    def bounds(self, partition: int) -> Tuple[T, T]:
        """
        Given a partition, return bounds if the partition
        :param partition:
        :return:
        """
        raise NotImplementedError()


class Log2Partitioner(Partitioner[float]):
    # TODO: this can be improved to allow for using k and min value to decide on max value and vice versa
    def __init__(self, min_value: float = 1, max_value: float = 2 ** 8):
        self.min_value = min_value if min_value is not None else 0
        self.max_value = max_value if max_value is not None else 2 ** 8

        self.k = int(np.log2(max_value)) - int(np.log2(min_value)) + 1  # adding to include smaller and bigger than boundaries

    def partition(self, value: float) -> int:
        if value <= self.min_value:
            return 0

        if value > self.max_value:
            return int(np.log2(self.max_value))

        return int(np.log2(value))

    def bounds(self, partition: int) -> Tuple[float, float]:
        if partition < 0 or partition > self.k:
            raise ValueError(f'invalid partition given {partition}')

        return 2 ** (partition - 1), 2 ** partition


class PartitionedList:
    _values: List[float]
    _partition_indices: List[int]

    def __init__(self, partitioner: Partitioner):
        self._values = list()
        self.partitioner = partitioner
        self.k = partitioner.k
        self._partition_indices = [0] * partitioner.k

    def __add__(self, v):
        p = self.partitioner.partition(v)
        l_index = self._partition_indices[p]
        r_index = self._partition_indices[p + 1] if p < self.k else len(self._values)
