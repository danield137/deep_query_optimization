from dqo.tree import Tree, Node


def test_pprint():
    t = Tree()
    '''
                        1
                  2            3
             4       5      7      8
          9    10      
    '''
    t.root = Node(1, children=[
        Node(2, children=[
            Node(4, children=[
                Node(9),
                Node(10)
            ]),
            Node(5)
        ]),
        Node(3, children=[
            Node(7),
            Node(8)
        ])
    ])

    pretty = t.pretty()

    assert pretty == """- 1
  - 2
    - 4
      - 9
      - 10
    - 5
  - 3
    - 7
    - 8
"""


def test_depth():
    t = Tree()
    t.root = Node(1, children=[
        Node(2, children=[
            Node(4, children=[
                Node(9),
                Node(10)
            ]),
            Node(5)
        ]),
        Node(3, children=[
            Node(7),
            Node(8)
        ])
    ])
    assert t.depth() == 3


def test_cloning():
    t = Tree()
    t.root = Node(1, children=[Node(2, children=[Node(4, children=[Node(9), Node(10)]), Node(5)]), Node(3, children=[Node(7), Node(8)])])
    import copy
    cloned = copy.deepcopy(t)

    cloned.root.children[0].value = 22

    assert cloned.root.children[0].value == 22
    assert t.root.children[0].value == 2


def test_path():
    t = Tree()
    t.root = Node(1, children=[Node(2, children=[Node(4, children=[Node(9), Node(10)]), Node(5)]), Node(3, children=[Node(7), Node(8)])])
    import copy
    cloned = copy.deepcopy(t)
    node = cloned.root.children[0].children[1]
    assert t.node_at(node.path()).value == node.value


def test_swap():
    t = Tree()
    t.root = Node(1, children=[Node(2, children=[Node(4, children=[Node(9), Node(10)]), Node(5)]), Node(3, children=[Node(7), Node(8)])])

    t.swap_nodes(t.root[0][1], t.root[1])

    assert t.root[1].value == 5
    assert t.root[0][1].value == 3


def test_permutations():
    t = Tree()
    t.root = Node(1, children=[Node(2, children=[Node(4, children=[Node(9), Node(10)]), Node(5)]), Node(3, children=[Node(7), Node(8)])])

    permuts = t.permutations()

    assert len(permuts) == 16


def test_preorder():
    t = Tree()
    t.root = Node(1, children=[
        Node(2, children=[
            Node(4, children=[
                Node(9),
                Node(10)
            ]),
            Node(5)
        ]),
        Node(3, children=[
            Node(7),
            Node(8)
        ])
    ])
    assert [n.value for n in t.preorder()] == [1, 2, 4, 9, 10, 5, 3, 7, 8]


def test_postorder():
    t = Tree()
    t.root = Node(1, children=[
        Node(2, children=[
            Node(4, children=[
                Node(9),
                Node(10)
            ]),
            Node(5)
        ]),
        Node(3, children=[
            Node(7),
            Node(8)
        ])
    ])
    assert [n.value for n in t.postorder()] == [9, 10, 4, 5, 2, 7, 8, 3, 1]


def test_inorder():
    t = Tree()
    t.root = Node(1, children=[
        Node(2, children=[
            Node(4, children=[
                Node(9),
                Node(10)
            ]),
            Node(5)
        ]),
        Node(3, children=[
            Node(7),
            Node(8)
        ])
    ])
    assert [n.value for n in t.inorder()] == [9, 4, 10, 2, 5, 1, 7, 3, 8]


def test_bfs():
    t = Tree()
    t.root = Node(1, children=[
        Node(2, children=[
            Node(4, children=[
                Node(9),
                Node(10)
            ]),
            Node(5)
        ]),
        Node(3, children=[
            Node(7),
            Node(8)
        ])
    ])
    assert [n.value for n in t.bfs()] == [1, 2, 3, 4, 5, 7, 8, 9, 10]


def test_transform():
    a = Tree()
    a.root = Node(1, children=[Node(2, children=[Node(4, children=[Node(9), Node(10)]), Node(5)]), Node(3, children=[Node(7), Node(8)])])
    b = Tree.transform(a, lambda n: Node(n.value * 2))
    assert [n.value for n in b.dfs()] == [n.value * 2 for n in a.dfs()]
