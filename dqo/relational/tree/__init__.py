from __future__ import annotations

import copy
import dataclasses
import logging
from collections import defaultdict
from dataclasses import field
from typing import List, Dict, cast, Set, Optional, Tuple

from dqo.relational.tree.node import RelationColumn, RelationalNode, RelationNode, AliasNode, SelectionNode, JoinNode, ProjectionNode, \
    OrNode, Operand
from dqo.tree import Tree

logger = logging.getLogger(__name__)


class SqlSyntaxError(Exception):
    pass


class MissingFromClauseError(SqlSyntaxError):
    pass


def propose_pushdown_locations(selection: SelectionNode) -> Tuple[List[RelationalNode], List[SelectionNode]]:
    """
    Search for possible locations that are better than the existing one.
    It can happen under these scenarios:
    1. Joins, where selection can be pushed under the join, and closer to the actual relations.
    2. Projections, where selection can be pushed down under the projection.
    3. Multiple spread out selections, where selections can be merged.
    :param selection:
    :return: pushables (List), mergeables (List)
        lists, because a single selection can be pushed down to multiple descendants.
        consider a case where a single table is viewed in multiple ways under a single join.
    """
    # check if selection applies to only a subset of the children,
    # if so, push it closer to the relation

    if not set(selection.relations()) or isinstance(selection, JoinNode):
        return [], []

    rel_operand: RelationColumn = selection.relational_operands()[0]
    deepest_relation = rel_operand.deepest_relation()

    pushables, mergeables = [], []
    s = selection.children[:]
    while len(s) > 0:
        n = s.pop()
        if type(n) is SelectionNode:
            if rel_operand.same_entity(n.relational_operands()[0]):
                mergeables.append(cast(SelectionNode, n))
                break
        elif isinstance(n, RelationNode):
            if deepest_relation == n:
                pushables.append(cast(RelationNode, n))
                break

        for c in n.children:
            s.append(c)

    return pushables, mergeables


def merge_selections_(a: SelectionNode, b: SelectionNode) -> bool:
    """
    Merges a into b.
    :param a:
    :param b:
    :return: returns the merged Selections if possible to merge, otherwise returns None.
    """
    a_operand = next(filter(lambda o: type(o) is Operand, a.operands))
    b_operand = next(filter(lambda o: type(o) is Operand, b.operands))

    if a_operand.is_num():
        if any('=' in op for op in [a.operator, b.operator]):
            if a.operator == '=' or b.operator == '=':
                # if a = x and b = y
                if a.operator == b.operator:
                    # if x = y
                    if a_operand.text == b_operand.text:
                        return True
                # if a = x and b >= y
                elif a.operator == '=':
                    if (
                            b.operator == '>=' and float(a_operand.text) >= float(b_operand.text)
                    ) or (
                            b.operator == '<=' and float(a_operand.text) <= float(b_operand.text)
                    ):
                        b.operator = '='
                        b.operands[1].text = a_operand.text
                        return True
                # if a >= x and b = y
                else:
                    if (
                            a.operator == '>=' and float(b_operand.text) >= float(a_operand.text)
                    ) or (
                            a.operator == '<=' and float(b_operand.text) <= float(a_operand.text)
                    ) or (
                            a.operator == '<' and float(b_operand.text) < float(a_operand.text)
                    ) or (
                            a.operator == '>' and float(b_operand.text) > float(a_operand.text)
                    ):
                        return True

            # if a >= x and b >= y
            elif a.operator == b.operator:
                if (
                        a.operator == '>=' and float(a_operand.text) >= float(b_operand.text)
                ) or (
                        a.operator == '>' and float(a_operand.text) < float(b_operand.text)
                ) or (
                        a.operator == '<=' and float(a_operand.text) <= float(b_operand.text)
                ) or (
                        a.operator == '<' and float(a_operand.text) > float(b_operand.text)
                ):
                    b.operands[1].text = a_operand.text

                return True
            else:
                # if a >= x and b <= x
                if a_operand.text == b_operand.text:
                    b.operator = '='
                    return True
        elif a.operator == b.operator:
            if (
                    a.operator == '>' and float(a_operand.text) > float(b_operand.text)
            ) or (
                    a.operator == '<' and float(a_operand.text) < float(b_operand.text)
            ) or (
                    a.operator == '<=' and float(a_operand.text) <= float(b_operand.text)
            ) or (
                    a.operator == '>=' and float(a_operand.text) >= float(b_operand.text)
            ):
                b.operands[1].text = a_operand.text
            return True

    return False


def _push_down_selections_(tree: RelationalTree):
    selections = tree.get_selections(include_joins=False)
    visited = set()
    for selection in selections:
        if selection in visited:
            continue

        visited.add(selection)

        if any(isinstance(operand, RelationColumn) for operand in selection.operands):
            pushables, mergeables = propose_pushdown_locations(selection)
            detach = False
            if pushables or mergeables:
                selection.detach()

                if pushables:
                    for pushable in pushables:
                        SelectionNode(operator=selection.operator, operands=selection.operands).push_above(pushable)

                if mergeables:
                    for mergeable in mergeables:
                        detach |= merge_selections_(selection, mergeable)


def prune_redundant_projections_(tree: RelationalTree):
    ...


def _canonize_(tree: RelationalTree):
    for n in tree.nodes():
        n.canonize_()


class RelationalTree(Tree[RelationalNode]):
    sql: str
    relations: Dict[str, RelationNode] = field(default_factory=dict)

    def __init__(self, sql: str, relations: Optional[Dict[str, RelationNode]] = None, root: Optional[RelationalNode] = None):
        super().__init__(root)

        self.sql = sql
        self.relations = relations

    def get_joins(self) -> List[JoinNode]:
        return cast(List[JoinNode], self.filter_nodes(lambda n: type(n) is JoinNode))

    def get_relations(self) -> List[RelationNode]:
        return cast(List[RelationNode], self.filter_nodes(lambda n: type(n) is RelationNode))

    def get_projections(self) -> List[ProjectionNode]:
        projections = cast(List[ProjectionNode], self.filter_nodes(lambda n: type(n) is ProjectionNode))

        return projections

    def get_selections(self, include_joins=True, flatten_or=False) -> List[SelectionNode]:
        result = []
        selections = cast(List[SelectionNode], self.filter_nodes(lambda n: isinstance(n, SelectionNode)))

        for selection in selections:
            if isinstance(selection, OrNode):
                if flatten_or:
                    for or_selection in selection.flatten_selections():
                        result.append(or_selection)
                else:
                    result.append(selection)
            elif not isinstance(selection, JoinNode):
                result.append(selection)
            elif include_joins:
                result.append(selection)

        return result

    def permutations(self, limit: Optional[int] = None) -> List[RelationalTree]:
        super_permutations = super().permutations(limit)

        if len(self.get_selections(include_joins=False)) == 0:
            return super_permutations

        extra_permutations = []

        for tree in super_permutations:
            so_far = len(super_permutations) + len(extra_permutations)
            if limit and 0 < limit <= so_far:
                return extra_permutations + super_permutations

            tree = cast(RelationalTree, tree)
            conditions = tree.get_selections(include_joins=False)

            not_visited = set(conditions)

            for condition in conditions:
                current = condition

                sequence_len = 1
                while current.children and isinstance(current[0], SelectionNode):
                    current = cast(SelectionNode, current[0])
                    if current in not_visited:
                        not_visited.remove(current)
                        sequence_len += 1

                if sequence_len > 1:
                    from sympy.utilities.iterables import multiset_permutations
                    for permuted_indices in multiset_permutations(list(range(sequence_len))):
                        natural_order = list(range(sequence_len))
                        if permuted_indices != natural_order:
                            new_tree = copy.deepcopy(tree)
                            current = new_tree.node_at(condition.path())
                            nodes_in_seq = [current]
                            for i in range(sequence_len - 1):
                                current = current[0]
                                nodes_in_seq.append(current)

                            swaps = [(i, v) for i, v in enumerate(permuted_indices)]
                            while swaps:
                                old_position, new_position = swaps.pop(0)
                                if old_position == new_position:
                                    continue
                                new_tree.swap_nodes(nodes_in_seq[old_position], nodes_in_seq[new_position])
                                # update pointers of remaining swaps
                                for i, (_old_position, _new_position) in enumerate(swaps):
                                    if _old_position == new_position:
                                        swaps[i] = (old_position, _new_position)

                            extra_permutations.append(new_tree)

        return extra_permutations or super_permutations

    def get_selection_columns(self, exclude_aliases=False) -> List[RelationColumn]:
        columns: Set[RelationColumn] = set()
        selection_nodes = self.get_selections()

        if not selection_nodes:
            return []

        selection_node = selection_nodes.pop()
        while selection_node is not None:
            for operand in selection_node.operands:
                if isinstance(operand, RelationColumn):
                    if not exclude_aliases or not isinstance(cast(RelationColumn, operand).relation, AliasNode):
                        columns.add(cast(RelationColumn, operand))
                elif isinstance(operand, SelectionNode):
                    selection_nodes.append(operand)

            selection_node = selection_nodes.pop() if len(selection_nodes) > 0 else None

        return list(columns)

    def optimize(self):
        _canonize_(self)
        _push_down_selections_(self)

        # prune_redundant_projections_(self)

    def pretty(self):
        setattr(self.root, 'depth', 0)
        nodes = [self.root]

        relations = list(self.relations.values())
        joins = self.get_joins()
        selections = self.get_selection_columns(exclude_aliases=False)
        projections = self.get_projections()
        graphic_tree = ''
        while len(nodes) > 0:
            node = nodes.pop(0)
            graphic_tree += node.depth * '  ' + str(node) + '\n'
            if node.children:
                for c in node.children:
                    setattr(c, 'depth', getattr(node, 'depth') + 1)
                    nodes.insert(0, c)

        output = f'''
{"=" * 80}
{self.sql}
{'-' * 80}
Relations ({len(relations)}):
{relations}
Projections ({len(projections)}):
{projections}
Joins ({len(joins)}):
{joins}
Predicate Columns ({len(selections)}):
{selections}
{'-' * 80}
{graphic_tree}
{'=' * 80}'''

        logger.debug(output)
        return output

    def dangling(self):
        dangling = []
        for relation_node in self.relations.values():
            if id(relation_node.highest()) != id(self.root):
                dangling.append(relation_node)

        return dangling

    def get_join_graph(self, cache=True) -> JoinGraph:
        if cache and hasattr(self, '_g'):
            return getattr(self, '_g')

        g = JoinGraph()
        for join in self.get_joins():
            g.add_join(join)

        if cache:
            setattr(self, '_g', g)

        return g

@dataclasses.dataclass
class JoinGraphEdge:
    left_rel: str
    left_col: str
    right_rel: str
    right_col: str

    def __hash__(self) -> int:
        return hash(f"{self.left_rel}.{self.left_col}-{self.right_rel}.{self.right_col}")

@dataclasses.dataclass
class JoinGraphNode:
    name: str
    connections: Set[JoinGraphEdge] = field(default_factory=set)
    left_sided: Dict[str, Dict[str, Set[JoinGraphEdge]]] = field(default_factory=dict)
    right_sided: Dict[str, Dict[str, Set[JoinGraphEdge]]] = field(default_factory=dict)

    def join(self, other: JoinGraphNode, left_col: str, right_col: str) -> JoinGraphEdge:
        edge = JoinGraphEdge(self.name, left_col, other.name, right_col)

        if left_col not in self.left_sided:
            self.left_sided[left_col] = defaultdict(set)
        if right_col not in other.right_sided:
            other.right_sided[right_col] = defaultdict(set)

        self.left_sided[left_col][other.name].add(edge)
        other.right_sided[right_col][self.name].add(edge)

        self.connections.add(edge)
        other.connections.add(edge)

        return edge


@dataclasses.dataclass
class JoinGraph:
    edges: Set[JoinGraphEdge] = field(default_factory=set)
    nodes: Dict[str, JoinGraphNode] = field(default_factory=dict)

    def add_join(self, join: JoinNode):
        if join.left.name not in self.nodes:
            self.nodes[join.left.name] = JoinGraphNode(join.left.name)
        if join.right.name not in self.nodes:
            self.nodes[join.right.name] = JoinGraphNode(join.right.name)

        edge = self.nodes[join.left.name].join(self.nodes[join.right.name], join.operands[0].column, join.operands[1].column)
        self.edges.add(edge)

    def get_joins(self, a, b, ltr=False) -> Optional[List[JoinGraphEdge]]:
        if a not in self.nodes or b not in self.nodes:
            return

        a_node = self.nodes[a]
        connections = set()

        if a_node.connections:
            for connection in a_node.connections:
                if (
                        connection.left_rel == a and connection.right_rel == b or
                        connection.left_rel == b and connection.right_rel == a
                ):
                    connections.add(connection)

        return list(connections)

    # Naive path finding O(V*E) - start from a, try to get to b.
    def get_path(self, a, b) -> Optional[JoinGraphEdge]:
        a_node = self.nodes[a]
        b_node = self.nodes[b]

        stack = [(None, a_node)]
        visited = set()
        while stack:
            start, node = stack.pop()
            visited.add(node.name)
            # because multiple connections can lead to the same target, make sure add to add it only once
            local_visited = set()
            for c in node.connections:
                if c.left_rel == node.name:
                    other_name = c.right_rel
                    start = start or c.left_col
                    end = c.right_col
                else:
                    other_name = c.left_rel
                    start = start or c.right_col
                    end = c.left_col

                if other_name == b:
                    return JoinGraphEdge(a, start, b, end)
                if other_name not in visited and other_name not in local_visited:
                    local_visited.add(other_name)

                    stack.append((start, self.nodes[other_name]))

        return
