from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from typing import List, Optional, Union


@total_ordering
@dataclass
class TableRef:
    name: str
    alias: Optional[str] = None

    def __hash__(self):
        return hash(f'{self.__class__.__name__}_{self.alias or self.name}')

    def __eq__(self, other: ColumnRef):
        return self.name == other.name

    def __lt__(self, other: ColumnRef):
        return self.name < other.name


@total_ordering
@dataclass
class ColumnRef:
    name: str
    table: TableRef
    alias: Optional[str] = None

    @property
    def full_name(self) -> str:
        name = self.alias or self.name

        if self.table:
            if self.table.alias:
                return f'{self.table.alias}.{name}'

            return f'{self.table.name}.{name}'

        return name

    def __str__(self):
        return self.full_name

    def __hash__(self):
        return hash(f'{self.__class__.__name__}_{self.full_name}')

    def __eq__(self, other: ColumnRef):
        return self.full_name == other.full_name

    def __lt__(self, other: ColumnRef):
        return self.full_name < other.full_name


@dataclass
class Const:
    text: str

    def __str__(self):
        return self.text

    def __hash__(self):
        return hash(self.text)


@total_ordering
@dataclass
class Projection:
    # TODO: at this point I'm assuming projection are only column based
    col: ColumnRef
    func: Optional[str] = None

    def __hash__(self):
        return hash(self.col if self.func is None else hash(self.col) + hash(self.func))

    def __str__(self):
        if self.func:
            return f'{self.func}({self.col})'
        return str(self.col)

    def __eq__(self, other: ColumnRef):
        return str(self) == str(other)

    def __lt__(self, other: ColumnRef):
        return str(self) < str(other)


@total_ordering
@dataclass
class Selection:
    operator: str
    operands: List[Union[Const, ColumnRef]]

    def __hash__(self):
        return hash(self.operator) + sum([hash(o) for o in self.operands])

    def __str__(self):
        return f'{self.operator}:{",".join([ str(o) for o in self.operands])}'

    def __eq__(self, other: ColumnRef):
        return str(self) == str(other)

    def __lt__(self, other: ColumnRef):
        return str(self) < str(other)


@total_ordering
@dataclass
class Condition(Selection):
    col: ColumnRef
    value: Const

    def __init__(self, col: ColumnRef, value: Const, operator: str, ltr: bool = True):
        super().__init__(operator, [col, value])
        self.col = col
        self.value = value
        self.ltr = ltr

    def __str__(self):
        left = self.col.full_name if self.ltr else self.value
        right = self.value if self.ltr else self.col.full_name
        return f'{left} {self.operator} {right}'

    def __hash__(self):
        return hash(self.col) + hash(self.value) + hash(self.operator)

    def __eq__(self, other: ColumnRef):
        return str(self) == str(other)

    def __lt__(self, other: ColumnRef):
        return str(self) < str(other)

@total_ordering
@dataclass
class Join(Selection):
    left: ColumnRef
    right: ColumnRef

    def __init__(self, left: ColumnRef, right: ColumnRef, operator: str):
        super().__init__(operator, [left, right])
        self.left = left
        self.right = right

    def __str__(self):
        return f'{self.left.full_name} {self.operator} {self.right.full_name}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: ColumnRef):
        return str(self) == str(other)

    def __lt__(self, other: ColumnRef):
        return str(self) < str(other)
