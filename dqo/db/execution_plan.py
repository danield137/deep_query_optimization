import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dqo.tree import Tree, Node


@dataclass
class ExecutionOp(Node):
    def __init__(self, value: str, parent: Optional[Node] = None, children: Optional[List[Node]] = None):
        super(ExecutionOp, self).__init__(value, parent, children)

    def __str__(self):
        return f"{self.__class__.__name__}({self.value})"

    @staticmethod
    def parse(json_result: dict) -> Optional['ExecutionOp']:
        node_type = json_result["Node Type"]
        if node_type == 'Aggregate':
            return AggregateOp()
        if node_type.endswith('Scan'):
            scan_type = node_type[:-4].strip()
            parallel = False

            if scan_type.startswith('Parallel'):
                scan_type = scan_type.split('Parallel')[0].strip()
                parallel = True

            return ScanOp(
                scan_type,
                json_result.get('Filter') or json_result.get('Index Cond'),
                json_result.get('Relation Name'),
                parallel
            )
        if node_type.endswith('Join'):
            join_type = node_type[:-4].strip()
            parallel = False

            if join_type.startswith('Parallel'):
                join_type = join_type.split('Parallel')[0].strip()
                parallel = True

            condition_props = [k for k in json_result.keys() if k.lower().endswith('condition') or k.lower().endswith('filter') or k.lower().endswith('cond')]

            condition = None
            for condition_prop in condition_props:
                condition = json_result[condition_prop]
                if condition:
                    break

            return JoinOp(join_type, condition, parallel)

        if node_type == 'Nested Loop':
            # TODO: check what are nested loop joins
            return NestedLoopOp(json_result.get('Join Type', None), json_result.get('Join Filter', None))

        return


@dataclass
class AggregateOp(ExecutionOp):
    def __init__(self):
        super().__init__("", children=[])


@dataclass
class JoinFilter:
    left_rel: str
    left_col: str
    right_rel: str
    right_col: str

    @staticmethod
    def parse(string) -> Optional['JoinFilter']:
        if not string:
            return None

        try:
            string = string.replace('(', '').replace(')', '')
            left, right = None, None

            string = string.replace('IS NOT NULL', '<> NULL').replace('IS NULL', '= NULL')

            # TODO: improve this when needed
            splitters = ['<>', '<=', '>=', '=', '>', '<', '~~']
            for splitter in splitters:
                if splitter in string:
                    left, right = string.split(splitter)
                    break

            left_rel, left_col, right_rel, rel_col = None, None, None, None
            if left:
                if '.' in left:
                    left_rel, left_col = [s.strip() for s in left.split('.')]
                else:
                    left_col = left.strip()

            left_col = left_col.split('::')[0]

            if right:
                if '.' in right:
                    right_rel, right_col = [s.strip() for s in right.split('.')]
                else:
                    right_col = right.strip()

            right_col = right_col.split('::')[0]
            return JoinFilter(left_rel, left_col, right_rel, right_col)
        except Exception as ex:
            print(ex)
            raise ValueError(f"Can't parse '{string}', error={ex}")


@dataclass
class Condition:
    left_col: str
    right_col: str
    left_rel: Optional[str]
    right_rel: Optional[str]

    @staticmethod
    def parse(string) -> Optional['Condition']:
        if not string:
            return None

        try:
            string = string.replace('(', '').replace(')', '')
            left, right = None, None

            string = string.replace('IS NOT NULL', '<> NULL').replace('IS NULL', '= NULL')

            # TODO: improve this when needed
            splitters = ['<>', '<=', '>=', '=', '>', '<', '~~']
            for splitter in splitters:
                if splitter in string:
                    left, right = string.split(splitter)
                    break

            left_rel, left_col, right_rel, rel_col = None, None, None, None
            if left:
                if '.' in left:
                    left_rel, left_col = [s.strip() for s in left.split('.')]
                else:
                    left_col = left.strip()

            left_col = left_col.split('::')[0]

            if right:
                if '.' in right:
                    right_rel, right_col = [s.strip() for s in right.split('.')]
                else:
                    right_col = right.strip()

            right_col = right_col.split('::')[0]
            return Condition(left_col, right_col, left_rel, right_rel)
        except Exception as ex:
            print(ex)
            raise ValueError(f"Can't parse '{string}', error={ex}")




@dataclass
class ScanOp(ExecutionOp):
    # Hash, Index, Seq, Bitmap Hash
    scan_type: str
    scan_filter: Optional[List[Condition]]
    relation: str
    parallel: bool = False

    def __init__(self, scan_type: str, scan_filter: str, relation: str, parallel: bool = False):
        super().__init__(f'{scan_type}:{relation}:filter={scan_filter if scan_filter else ""}')
        self.scan_type = scan_type
        if scan_filter:
            self.scan_filter = [Condition.parse(s) for s in scan_filter.split(' AND ')]
        else:
            self.scan_filter = None
        self.relation = relation
        self.parallel = parallel


@dataclass
class JoinOp(ExecutionOp):
    # Hash, Index, Seq, Bitmap Hash
    join_type: str
    join_condition: Optional[List[Condition]]
    parallel: bool = False

    def __init__(self, join_type: str, join_condition: str, parallel: bool = False):
        super().__init__(join_condition or "")
        self.join_type = join_type
        if join_condition:
            self.join_condition = [Condition.parse(s) for s in join_condition.split(' AND ')]
        else:
            self.join_condition = None
        self.parallel = parallel
        self.children = []


@dataclass
class NestedLoopOp(ExecutionOp):
    join_type: Optional[str] = None
    join_filter: Optional[List[JoinFilter]] = None

    def __init__(self, join_type: str = None, join_filter: str = None):
        super().__init__(join_filter or "")
        self.join_type = join_type
        if join_filter:
            self.join_filter = [JoinFilter.parse(s) for s in join_filter.split(' AND ')]
        else:
            self.join_filter = None


class ExecutionPlan:
    @staticmethod
    def parse(json_result: dict) -> Tree:
        tree = Tree()

        stack: List[Tuple[Optional[ExecutionOp], dict]] = [(None, json_result)]

        while stack:
            parent, current = stack.pop()
            try:
                node = ExecutionOp.parse(current)

                if parent is None:
                    tree.root = node
                elif node:
                    parent.children = parent.children or []
                    node.set_parent(parent)
                # skip or merge non-important node (like gather)
                else:
                    if isinstance(parent, JoinOp) and parent.join_type.lower() == 'hash':
                        if 'Condition' in current and 'Condition' not in parent:
                            parent.join_condition = current['Condition']
                    node = parent

                if current.get('Plans'):
                    for p in current['Plans']:
                        stack.append((node, p))
            except Exception as ex:
                print(ex, current)
                raise ValueError(f'Cant handle {current}, error={ex}')

        return tree
