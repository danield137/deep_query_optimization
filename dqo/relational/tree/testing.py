from enum import Enum

from relational import RelationalTree


class EqualityMode(Enum):
    ISOMORPHIC = 'isomorphic'
    STRUCTURE = 'structure'
    CONTENT = 'content'


def assert_equal(a: RelationalTree, b: RelationalTree, eq_mode: EqualityMode = EqualityMode.CONTENT):
    if eq_mode == EqualityMode.ISOMORPHIC:
        # sort both trees first
        raise NotImplementedError()

    a_stack = [a.root]
    b_stack = [b.root]

    while a_stack or b_stack:
        a_curr = a_stack.pop()
        b_curr = b_stack.pop()

        if type(a_curr) != type(b_curr):
            raise AssertionError(f'Expected {type(a_curr)}, got {type(b_curr)}')

        if eq_mode == EqualityMode.CONTENT:
            if str(a_curr) != str(b_curr):
                raise AssertionError(f'Expected {a_curr}, got {b_curr}')

        if a_curr.children:
            for c in a_curr.children:
                a_stack.append(c)
        if b_curr.children:
            for c in b_curr.children:
                b_stack.append(c)

    if len(a_stack) > 0 or len(b_stack) > 0:
        raise AssertionError('Finished traversing trees, but they are not of same depth.')
