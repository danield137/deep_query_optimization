import inspect
import uuid
from typing import Optional, Generic, TypeVar, List, Iterator, Callable, Tuple
from collections import deque
T = TypeVar('T')


class Node(Generic[T]):
    value: T
    parent: Optional['NodeType'] = None
    children: Optional[List['NodeType']] = None

    def __init__(self, value: T = None, parent: Optional['NodeType'] = None, children: Optional[List['NodeType']] = None):
        self.value = value
        self.children = children or []
        self.parent = parent

        if parent is not None:
            parent.children.append(self)

        if children is not None:
            for child in children:
                child.parent = self

        self._uid = str(uuid.uuid4())

    def __getitem__(self, item: int):
        return self.children[item]

    def merge(self, other: 'NodeType'):
        self.value = other.value

    def path(self):
        n = self
        path = ''
        while n.parent:
            index = n.parent.children.index(n)
            n = n.parent
            path = str(index) + '.' + path if path else str(index)
        return '$.' + path if path else '$'

    def push_above(self, node: 'NodeType'):
        if node.parent:
            node.parent.children = [c for c in node.parent.children if c != node] + [self]
        self.parent = node.parent
        node.parent = self

        self.children.append(node)

    def push_under(self, node: 'NodeType'):
        self.set_children(node.children)
        self.parent = node
        node.children = [self]

    def detach_parent(self) -> Optional['NodeType']:
        p = self.parent
        if p is not None:
            self.parent = None
            p.children = [c for c in p.children if c != self]

        return p

    def detach_children(self) -> List['NodeType']:
        detached = []
        if self.children:
            for c in self.children:
                c.parent = None
                detached.append(c)
        self.children = []
        return detached

    def detach(self) -> List['NodeType']:
        parent = self.parent
        index_in_parent = -1
        if parent is not None:
            self.parent = None
            index_in_parent = parent.children.index(self)
            parent.children = [c for c in parent.children if c != self]

        children = self.detach_children()
        if parent is not None:
            for c in children:
                parent.children.insert(index_in_parent, c)
                c.parent = parent

        return children

    def set_parent(self, parent: 'NodeType'):
        if self.parent:
            self.parent.children = [c for c in self.parent.children if c != self]

        self.parent = parent
        if self not in parent.children:
            parent.children.append(self)

    def set_children(self, children: List['NodeType']):
        for child in children:
            if child.parent is not None:
                child.parent.children = [c for c in child.parent.children if c != child]
            child.parent = self

        self.children = children

    def reorder_children(self, new_indices: List[int]):
        self.children = [self.children[i] for i in new_indices]

    def intersection(self, other, exclude_self=False) -> Optional['NodeType']:
        p1 = self
        if exclude_self:
            if p1.parent is not None and other.parent is not None:
                p1 = p1.parent
                other = other.parent
            else:
                return None
        while p1 is not None:
            p2 = other
            while p2 is not None:
                if p1 == p2:
                    return p1
                p2 = p2.parent

            p1 = p1.parent

        return None

    def ancestors(self) -> Iterator['NodeType']:
        p = self

        while p.parent is not None:
            p = p.parent
            yield p

    def highest(self, must: Optional[Callable[['NodeType'], bool]] = None) -> 'NodeType':
        result = None
        for a in self.ancestors():
            if must and callable(must) and not must(a):
                break
            result = a
        return result

    def preorder(self):
        return self.descendents('dfs')

    def inorder(self):
        current = self
        s = []
        while current or len(s) > 0:
            if current is not None:
                s.append(current)
                current = current.children[0] if current.children else None
            elif s:
                current = s.pop()
                yield current
                current = current.children[1] if len(current.children) == 2 else None

    def postorder(self):
        def peek(s):
            if len(s) > 0:
                return s[-1]
            return None

        stack = []
        current = self
        while True:
            while current:
                if current.children and len(current.children) == 2:
                    stack.append(current.children[1])
                stack.append(current)

                current = current.children[0] if current.children else None

            current = stack.pop()

            if current.children and len(current.children) == 2 and id(peek(stack)) == id(current.children[1]):
                stack.pop()
                stack.append(current)
                current = current.children[1]
            else:
                yield current
                current = None

            if len(stack) <= 0:
                break

    def descendents(self, strategy='bfs', ghost_children=False):
        """
            In order BFS
            :return:
        """
        q = [self]

        while len(q) > 0:
            node = q.pop(0 if strategy == 'bfs' else -1)
            yield node

            if node is not None:
                if node.children:
                    children = node.children if strategy == 'bfs' else node.children[::-1]
                    for c in children:
                        q.append(c)

                    if ghost_children and len(node.children) == 1:
                        q.append(None)

    def leaves(self) -> Tuple[int, 'NodeType']:
        q = [(0, self)]
        while len(q) > 0:
            lvl, node = q.pop(0)
            if node.children:
                for c in node.children:
                    q.append((lvl + 1, c))
            else:
                yield lvl, node

    def head(self, include_self=True) -> Optional['NodeType']:
        ancestors = list(self.ancestors())

        if ancestors:
            return ancestors[-1]
        elif include_self:
            return self

        return None

    def bubble_up(self):
        head = self.head(False)
        if head:
            self.detach()
            head.set_parent(self)

    def is_leaf(self) -> bool:
        return self.children is None or len(self.children) > 0


NodeType = TypeVar('NodeType', bound=Node)

TreeType = TypeVar('TreeType', bound='Tree')


class Tree(Generic[NodeType]):
    root: Optional[NodeType] = None

    def __init__(self, root: Optional[NodeType] = None):
        self._uid = str(uuid.uuid4())
        self.root = root

    def apply(self, func: Callable[[NodeType], None]):
        for node in self.nodes():
            func(node)

    def preorder(self):
        return self.root.preorder()

    def inorder(self):
        return self.root.inorder()

    def postorder(self):
        return self.root.postorder()

    def dfs(self, ghost_children=False) -> Iterator[NodeType]:
        return self.root.descendents(strategy='dfs', ghost_children=ghost_children)

    def bfs(self, ghost_children=False) -> Iterator[NodeType]:
        return self.root.descendents(strategy='bfs', ghost_children=ghost_children)

    def nodes(self) -> Iterator[NodeType]:
        return self.bfs()

    def find(self, uid):
        for n in self.bfs():
            if n._uid == uid:
                return n
        return None

    def node_at(self, path: str) -> NodeType:
        parts = path.split('.')
        node = None
        for part in parts:
            if part == '$':
                node = self.root
            else:
                node = node.children[int(part)]
        return node

    def detach_from_root(self, attach_to: NodeType):
        attach_to.push_under(self.root)
        attach_to.parent = None

    def filter_nodes(self, condition: Optional[Callable[[NodeType], bool]]) -> List[NodeType]:
        if not hasattr(self, '_filtered_nodes'):
            setattr(self, '_filtered_nodes', {})

        condition_source = inspect.getsource(condition).strip() if condition else None
        conditional_nodes = getattr(self, '_filtered_nodes')
        if condition_source not in conditional_nodes:
            nodes = []
            for n in self.bfs():
                if not condition or condition(n):
                    nodes.append(n)

            conditional_nodes[condition_source] = nodes

        return conditional_nodes[condition_source]

    # TODO: do this faster
    def __len__(self):
        return len(list(self.dfs()))

    def depth(self) -> int:
        d = 0

        for leaf_depth, _ in self.root.leaves():
            if leaf_depth > d:
                d = leaf_depth

        return d

    def nodes_at_level(self, lvl) -> List[NodeType]:
        q = [(0, self.root)]
        while len(q) > 0:
            current_lvl, node = q.pop(0)
            if current_lvl == lvl:
                yield node
            elif current_lvl < lvl:
                if node.children:
                    for c in node.children:
                        q.append((current_lvl + 1, c))

    def swap_nodes(self, a: NodeType, b: NodeType):
        b_parent, a_parent = b.parent, a.parent
        b_children, a_children = b.children[:], a.children[:]

        if a.parent == b:
            a.set_parent(b.parent)
            b.set_parent(a)

            b.set_children(a_children)
        elif b.parent == a:
            b.set_parent(a.parent)
            a.set_parent(b)

            a.set_children(b_children)
        else:
            b.set_parent(a_parent)
            a.set_parent(b_parent)

            b.set_children(a_children)
            a.set_children(b_children)

    def permutations(self, limit: Optional[int] = None) -> List[TreeType]:
        from sympy.utilities.iterables import multiset_permutations
        import copy

        trees = [self]
        all_nodes = list(self.dfs())

        for n in all_nodes:
            new_trees = []
            for tree in trees:
                if limit and 0 < limit <= len(trees):
                    return trees
                uid = n._uid
                current = tree.find(uid)
                variations = []
                if hasattr(current, 'variations'):
                    variations = getattr(current, 'variations')()

                if current.children and len(current.children) > 1:
                    for perm_indices in multiset_permutations(list(range(len(current.children)))):
                        if variations:
                            for variation in variations:
                                new_tree = copy.deepcopy(tree)
                                new_node = new_tree.node_at(current.path())
                                new_node.reorder_children(perm_indices)
                                new_node.merge(variation)

                                new_trees.append(new_tree)
                        # exclude the natural sort
                        if perm_indices != list(range(len(perm_indices))):
                            new_tree = copy.deepcopy(tree)
                            new_node = new_tree.node_at(current.path())
                            new_node.reorder_children(perm_indices)

                            new_trees.append(new_tree)
                elif variations:
                    # TODO: this is ugly, every variation is essentially a split
                    for variation in variations:
                        new_tree = copy.deepcopy(tree)
                        new_node = new_tree.node_at(current.path())
                        new_node.merge(variation)

                        new_trees.append(new_tree)
            trees += new_trees
        return trees

    def pretty(self, value_only=True):
        nodes = [self.root]

        graphic_tree = ''
        while len(nodes) > 0:
            node = nodes.pop()
            padding = getattr(node, 'depth', 0) * '  '
            graphic_tree += f"{padding}- {str(node.value) if value_only else str(node)}\n"
            if node.children:
                for c in node.children[::-1]:
                    setattr(c, 'depth', getattr(node, 'depth', 0) + 1)
                    nodes.append(c)

        return graphic_tree

    @staticmethod
    def transform(a: 'Tree', node_transform: Callable[[Node], Node]) -> 'Tree':
        b = Tree()
        a2b = {}
        for a_node in a.dfs():
            a_node_id = id(a_node)

            b_node = node_transform(a_node)

            a2b[a_node_id] = b_node

            if a_node.parent is None:
                b.root = b_node
            else:
                b_parent = a2b[id(a_node.parent)]
                b_node.set_parent(b_parent)

        return b
