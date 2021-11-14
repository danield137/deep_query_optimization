from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Optional, List, Dict, cast, Union

from dqo.db.serializeable import Serializable
from dqo.relational.models import ColumnRef, TableRef


class DataType(str, Enum):
    FLOAT = 'float'
    NUMBER = 'number'
    STRING = 'string'
    BOOL = 'bool'
    TIME = 'time'


@dataclass
class ValueStats(Serializable):
    def __hash__(self):
        return hash(str(id(self)))


@dataclass
class NumericStats(ValueStats):
    min: float
    mean: float
    max: float

    variance: float
    skewness: float
    kurtosis: float

    hist: List[float]
    freq: List[int]


@dataclass
class StringStats(ValueStats):
    length: NumericStats
    word: NumericStats


@dataclass
class ColumnStats(Serializable):
    total: int
    nulls: int
    distinct: int
    index: bool = False
    values: Optional[Union[NumericStats, StringStats]] = None

    @property
    def nulls_fraction(self) -> float:
        return 1.0 * self.nulls / self.total if self.total > 0 else 0

    @property
    def distinct_ratio(self) -> float:
        real_values = self.total - self.nulls
        return 1.0 * self.distinct / real_values if real_values > 0 else 0

    def __hash__(self):
        return hash(f'{self.total}_{self.nulls}_{self.distinct}_{self.index}_{id(self)}')


@dataclass
class TableStats(Serializable):
    rows: Optional[int]
    pages: Optional[int]
    page_size: Optional[int]

    @property
    def size(self) -> int:
        return self.pages * (self.page_size or 8 * 1024)

    def __hash__(self):
        return hash(f'{self.__class__.__name__}_{self.rows}_{self.pages}_{self.page_size}')


@total_ordering
@dataclass
class Column(Serializable):
    name: str
    data_type: DataType
    stats: Optional[ColumnStats] = field(default=None, repr=False)

    def __init__(self, name, data_type, stats=None):
        self.name = name
        self.data_type = data_type if isinstance(data_type, DataType) == str else DataType(data_type)
        self.stats = stats

    # TODO: this can be reflected away
    @classmethod
    def get_casts(cls):
        return [DataType]

    @classmethod
    def get_forward_refs(cls):
        return {
            'DataType': DataType,
            'Optional': Optional,
            'ColumnStats': ColumnStats
        }

    @property
    def full_name(self) -> str:
        if hasattr(self, 'table'):
            if self.table.alias:
                return f'{self.table.alias}.{self.name}'
            else:
                return f'{self.table.name}.{self.name}'
        else:
            return self.name

    def __repr__(self):
        return f'Column({self.name}, {self.data_type})'

    def __hash__(self):
        return hash(f'{self.__class__.__name__}_{self.name}_{self.data_type}_{hash(self.stats)}')

    def to_ref(self) -> ColumnRef:
        if not hasattr(self, 'table'):
            raise RuntimeError('expected column to be attached to a table. cant make a ref otherwise')

        t = cast(Table, getattr(self, 'table'))
        return ColumnRef(self.name, table=t.to_ref())

    def __lt__(self, other: Column) -> bool:
        return self.full_name < other.full_name

    def __eq__(self, other: Column) -> bool:
        return self.__repr__() == other.__repr__()


@dataclass
class Table(Serializable):
    name: str
    columns: List[Column]

    alias: Optional[str] = field(default=None)
    stats: Optional[TableStats] = field(default=None, repr=False)

    def __init__(self, name: str, columns: List[Column], **kwargs):
        self.name = name

        # bind to self
        for col in columns:
            col.table = self

        self.columns = columns

        self.group_by_type: bool = kwargs.get('group_by_type', True)

        self.columns_lookup: Dict[str, Column] = {}
        if self.group_by_type:
            self.types_lookup: Dict[DataType, List[Column]] = {}

        for column in columns:
            self.columns_lookup[column.name] = column
            if self.group_by_type:
                if column.data_type not in self.types_lookup:
                    self.types_lookup[column.data_type] = []
                self.types_lookup[column.data_type].append(column)

        self.alias = kwargs.get('alias', None)
        self.stats = kwargs.get('stats', None)

    def __post_init__(self):
        for col in self.columns:
            # TODO: fix this better
            self.columns_lookup[col.name] = col

    def __getstate__(self):
        state = self.__dict__.copy()

        del state['columns_lookup']
        if self.group_by_type:
            del state['types_lookup']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.columns_lookup = {}
        if self.group_by_type:
            self.types_lookup = {}

        for column in self.columns:
            self.columns_lookup[column.name] = column
            if self.group_by_type:
                if column.data_type not in self.types_lookup:
                    self.types_lookup[column.data_type] = []
                self.types_lookup[column.data_type].append(column)

    def __contains__(self, item):
        return item in self.columns_lookup

    def __getitem__(self, item):
        if type(item) == int:
            return self.columns[item]
        else:
            return self.columns_lookup[item]

    def __repr__(self):
        return f'Table({self.name}, {self.columns})'

    def __hash__(self):
        cols = '_'.join([f'{c.name}_{c.data_type.value}' for c in self.columns])
        return hash(f'{self.__class__.__name__}_{self.name}_{cols}')

    def to_ref(self) -> TableRef:
        return TableRef(self.name, self.alias)


@dataclass
class Database(Serializable):
    tables: List[Table]

    columns: List[Column] = field(init=False, repr=False)
    tables_lookup: Dict[str, Table] = field(repr=False, init=False)

    def __init__(self, tables: List[Table]):
        self.tables = tables
        self.tables_lookup = {}
        self.columns = []

        for i, table in enumerate(tables):
            self.tables_lookup[table.name] = table
            # TODO: this is duplicate code
            for col in table.columns:
                col.table = table

            self.columns += table.columns

    def __post_init__(self):
        self.columns = []
        # ensure ref to table
        for table in self.tables:
            # TODO: fix this better
            self.tables_lookup[table.name] = table
            for col in table.columns:
                col.table = table

            self.columns += table.columns

    def __hash__(self):
        table_names = '_'.join(sorted([t.name for t in self.tables]))
        return hash(f'{self.__class__.__name__}_{table_names}')

    @property
    def columns_count(self) -> int:
        return len(self.columns)

    def __contains__(self, item) -> bool:
        return item in self.tables_lookup

    def __getitem__(self, item: Union[int, str]) -> Table:
        """
        Given a key or index, returns table.
        """
        if type(item) == int:
            return self.tables[item]
        else:
            return self.tables_lookup[item]
