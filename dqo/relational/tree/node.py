from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Iterator, Union, cast, Tuple

from dqo.tree import Node


@dataclass
class Operand:
    text: str

    def __hash__(self):
        return hash(self.text)

    def is_num(self):
        return self.is_float() or self.is_int()

    def is_int(self):
        try:
            int(self.text)
            return True
        except:
            ...
        return False

    def is_float(self):
        try:
            float(self.text)
            return True
        except:
            ...
        return False

    def is_bool(self):
        return self.text in ['FALSE', 'TRUE']

    def duplicate(self) -> Operand:
        return Operand(self.text)


class MultiValueOperand(Operand):
    values: List[str]

    def __init__(self, values: List[str]):
        super().__init__(','.join(values))
        self.values = values

    def duplicate(self) -> MultiValueOperand:
        return MultiValueOperand(self.values)


@dataclass
class RelationColumn(Operand):
    relation: RelationNode
    column: str
    alias: Optional[str] = None
    aggregated: Optional[bool] = False
    func: Optional[str] = None

    @property
    def full_name(self):
        if self.relation:
            if self.relation.alias:
                return self.relation.alias + '.' + self.column
            return self.relation.name + '.' + self.column

        return self.column

    def deepest_relation(self) -> RelationNode:
        if type(self.relation) is RelationNode:
            return self.relation
        if type(self.relation) is AliasNode:
            root = self.relation
            if len(root.relations) == 1:
                return root.relations[0]

            s: List[Tuple[RelationalNode, str]] = [(c, self.column) for c in self.relation.children]

            while s:
                curr, col_name = s.pop()
                if isinstance(curr, ProjectionNode):
                    for col in curr.columns:
                        found = False
                        if (col.alias and col.alias == col_name) or (not col.alias and col.column == col_name):
                            if type(col.relation) is RelationNode:
                                return col.relation
                            elif type(col.relation) is AliasNode:
                                s.append((col.relation, col.column))
                            else:
                                raise RuntimeError()
                else:
                    for c in curr.children:
                        s.append((c, col_name))

    def same_entity(self, other: RelationColumn) -> bool:
        return self.column == other.column and other.deepest_relation() == self.deepest_relation()

    def __repr__(self):
        return f'RelationColumn(relation={self.relation}, column={self.column}, alias={self.alias})'

    def __hash__(self):
        return hash(self.text + '_' + self.column + '_' + (self.relation.name or self.relation.alias))

    def duplicate(self) -> RelationColumn:
        return RelationColumn(self.text, self.relation, self.column, self.alias, self.aggregated, self.func)


class RelationalNode(Node):
    children: List[RelationalNode]
    parent: Optional[RelationalNode] = None

    def __getitem__(self, item: int):
        return self.children[item]

    def join(self, left: RelationalNode, right: RelationalNode):
        children = [left.branch_head(), right.branch_head()]
        p = children[0].parent or children[1].parent

        if p is not None:
            self.set_parent(p)

        for c in children:
            c.set_parent(self)

    def branch_head(self) -> Optional[RelationalNode]:
        last = None
        for a in self.ancestors():
            # if a.parent is None:
            #     return last
            if (
                    isinstance(a, JoinNode) or
                    # currently, sub-selects are not supported. this means that the only projection node is the root.
                    # once this changes, need to add logic to bubble-up the correct projection (output) node.
                    # easiest way would probably to introduce a property (output=true) for top most projection.
                    (isinstance(a, ProjectionNode))
            ):
                break

            last = a

        return last or self

    def relations(self) -> Iterator[RelationNode]:
        return (d for d in self.descendents() if type(d) is RelationNode)

    def canonize_(self):
        raise NotImplementedError()

    @staticmethod
    def mock(children=None):
        return RelationalNode(children=children)


class RelationNode(RelationalNode):
    def __init__(self, name: str = None, alias: str = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.alias = alias

    def __repr__(self):
        return f'Relation(name={self.name}, alias={self.alias})'

    @staticmethod
    def mock(children=None):
        return RelationNode(children=children)

    def canonize_(self):
        ...

    def duplicate(self) -> RelationNode:
        return RelationNode(self.name, self.alias)


class AliasNode(RelationNode):
    def __init__(self, alias: str, relations: List[RelationNode], **kwargs):
        super().__init__(**kwargs)
        self.alias = alias
        self.relations = relations

    def __repr__(self):
        return f'Alias(alias={self.alias},relations={self.relations})'

    @staticmethod
    def mock(children=None):
        return AliasNode('', relations=[], children=children)

    def canonize_(self):
        ...


class OperatorVariations:
    flip = {
        '<': '>',
        '<=': '>=',
        '>=': '<=',
        '>': '<',
        '=': '=',
        '<>': '<>'
    }

    reorder_ops = set('IN')


class SelectionNode(RelationalNode):
    operator: str = None
    operands: List[Union[Operand, RelationalNode]] = None

    def __init__(self, operator: str = None, operands: List[Operand] = None, **kwargs):
        super().__init__(**kwargs)
        self.operator = operator
        self.operands = operands

    def merge(self, other: 'NodeType'):
        self.operator = other.operator
        self.operands = other.operands

    def variations(self):
        if type(self) is SelectionNode:
            if self.operator in OperatorVariations.flip:
                copied = copy.copy(self)
                copied.operands = copied.operands[::-1]
                copied.operator = OperatorVariations.flip[copied.operator]

                return [copied]

        return []

    def duplicate(self):
        return SelectionNode(self.operator, [o.duplicate() for o in self.operands])

    def relational_operands(self) -> List[RelationColumn]:
        return cast(List[RelationColumn], list(filter(lambda o: isinstance(o, RelationColumn), self.operands)))

    def related_relations(self) -> List[RelationNode]:
        operands = self.relational_operands()
        r: List[RelationNode] = []

        for operand in operands:
            # operand.relation
            if type(operand.relation) is AliasNode:
                for rel in operand.relation.relations:
                    r.append(rel)
            elif type(operand.relation) is RelationNode:
                r.append(operand.relation)
            else:
                raise RuntimeError()
        return r

    def __repr__(self):
        return f'Selection(operator="{self.operator}", operands={self.operands})'

    def canonize_(self):
        if type(self.operands[1]) is RelationColumn and type(self.operands[0]) is Operand:
            self.operands.reverse()
            self.operator = OperatorVariations.flip[self.operator]

    @staticmethod
    def mock(children=None):
        return SelectionNode(children=children, operands=[])


class OrNode(SelectionNode):
    # TODO: should replace left / right with a list of conditions
    left: RelationalNode
    right: RelationalNode

    def __init__(self, left: RelationalNode, right: RelationalNode, **kwargs):
        super().__init__('OR', [left, right], **kwargs)
        self.left = left
        self.right = right

    def flatten_selections(self) -> List[SelectionNode]:
        selections = []
        stack = [self.left, self.right]

        while len(stack) > 0:
            item = stack.pop()
            if isinstance(item, OrNode):
                stack.append(item.left)
                stack.append(item.right)
            else:
                selections.append(item)

        return selections

    def __repr__(self):
        return f'Or(left="{self.left}", right={self.right})'

    def set_parent(self, parent, detach_first=True):
        if detach_first and self.parent is not None:
            self.detach()
        self.parent = parent
        if self not in parent.children:
            parent.children.append(self)

    @staticmethod
    def mock(children=None):
        return OrNode(left=RelationalNode.mock(), right=RelationalNode.mock(), children=children)

    def canonize_(self):
        len_left = len(list(self.left.descendents()))
        len_right = len(list((self.right.descendents())))

        if len_left > len_right:
            self.left, self.right = self.right, self.left
        elif len_left == len_right:
            if type(self.left).__name__ > type(self.right).__name__:
                self.left, self.right = self.right, self.left
            elif type(self.left).__name__ == type(self.right).__name__:
                if str(self.left.value) > str(self.right.value):
                    self.left, self.right = self.right, self.left


class JoinNode(SelectionNode):
    def __init__(self, left: RelationalNode = None, right: RelationalNode = None, **kwargs):
        super().__init__(**kwargs)
        self.left = left
        self.right = right

        if left and right:
            intersection = left.intersection(right, exclude_self=True)

            if intersection:
                # check if the the intersection is an existing join
                if isinstance(intersection, JoinNode) and (
                        (intersection.left == left and intersection.right == right) or
                        (intersection.left == right and intersection.right == left)
                ):
                    self.push_above(intersection)
                else:
                    left_branch = left.highest(lambda n: n != intersection) or left
                    right_branch = right.highest(lambda n: n != intersection) or right

                    self.set_children([left_branch, right_branch])

                    # push join under intersection
                    self.parent = intersection
                    intersection.children.append(self)
            else:
                top_left = left.highest()
                top_right = right.highest()

                if getattr(top_right, 'out', False):
                    right = right.highest(lambda n: n != top_right) or right
                    left = top_left or left
                    self.set_children([left, right])
                    self.set_parent(top_right)
                elif getattr(top_right, 'out', False):
                    left = left.highest(lambda n: n != top_left) or left
                    right = top_right or right
                    self.set_children([left, right])
                    self.set_parent(top_left)
                else:
                    self.join(left, right)

    def reorder_children(self, new_indices: List[int]):
        super().reorder_children(new_indices)
        # FIXME: very hacky, should be done better
        moved = any([old_idx != new_idx for old_idx, new_idx in enumerate(new_indices)])
        if moved:
            self.left, self.right = self.right, self.left
            self.operands = [self.operands[idx] for idx in new_indices]

    def __repr__(self):
        return f'Join(left={self.left}, right={self.right})'

    def canonize_(self):
        super().canonize_()

        len_left = len(list(self.left.descendents()))
        len_right = len(list((self.right.descendents())))

        if len_left > len_right:
            self.left, self.right = self.right, self.left
        elif len_left == len_right:
            if type(self.left).__name__ > type(self.right).__name__:
                self.left, self.right = self.right, self.left
            elif type(self.left).__name__ == type(self.right).__name__:
                if str(self.left.value) > str(self.right.value):
                    self.left, self.right = self.right, self.left

    @staticmethod
    def mock(children=None):
        return JoinNode(children=children)


class ProjectionNode(RelationalNode):
    columns: List[RelationColumn]
    wildcard: bool = False
    out: bool = False

    def __init__(self, columns: List[RelationColumn] = None, wildcard=False, **kwargs):
        super().__init__(**kwargs)

        for column in columns:
            if column.relation is not None:
                if column.relation not in self.children:
                    self.children.append(column.relation)
                # TODO: not sure this is correct
                column.relation.parent = self

        self.columns = columns
        self.wildcard = wildcard

    def __repr__(self):
        return f'Projection(columns={self.columns}, astrix={self.wildcard})'

    def is_multi_relational(self):
        relations = set(c.relation for c in self.columns if isinstance(c, RelationColumn))
        return len(relations) > 1

    def canonize_(self):
        self.columns.sort(key=lambda x: x.full_name)

    @staticmethod
    def mock(children=None):
        return ProjectionNode([], children=children)

    def duplicate(self) -> ProjectionNode:
        dups = []
        dup_rels = {}
        for c in self.columns:
            if isinstance(c, RelationColumn):
                if c.relation.name not in dup_rels:
                    dup_rels[c.relation.name] = c.relation.duplicate()
                dups.append(RelationColumn(c.text, column=c.column, relation=dup_rels[c.relation.name], aggregated=c.aggregated, func=c.func))
        return ProjectionNode(dups, self.wildcard)


class AggregationNode(RelationalNode):
    def __init__(self, operator: str = None, field: str = None, relation: RelationNode = None, alias: str = None, **kwargs):
        super().__init__(**kwargs)

        if relation is not None:
            self.relation = relation
            self.children = [relation]
            relation.parent = self

        self.operator = operator
        self.field = field
        self.alias = alias

    def canonize_(self):
        ...

    def __repr__(self):
        return f'Aggregation(operator="{self.operator}", field={self.field},alias={self.alias})'
