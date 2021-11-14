from dqo.relational.tree import parser, RelationalTree, merge_selections_
from dqo.relational.tree.node import ProjectionNode, SelectionNode, RelationNode, AliasNode, JoinNode, Operand
from dqo.relational.tree.testing import assert_equal, EqualityMode


def test_tree_subselect_push_down_selection():
    query = """
        SELECT e.id, e.name 
        FROM (SELECT id,name from employees where salary < 100) as e  
        WHERE id > 1
        """

    actual = parser.parse_sql(query)
    print(actual.pretty())

    expected = RelationalTree(
        sql=query,
        root=ProjectionNode.mock([
            AliasNode.mock([
                ProjectionNode.mock([
                    SelectionNode.mock([
                        SelectionNode.mock([
                            RelationNode()
                        ])
                    ])
                ])
            ])

        ]))

    actual.optimize()
    print(actual.pretty())

    assert_equal(actual, expected, eq_mode=EqualityMode.STRUCTURE)


def test_selection_merge():
    test_data = [
        (
            (
                SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('1')]),
                SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('10')])
            ),
            SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('10')]),
            True
        ),
        (
            (
                SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('1')]),
                SelectionNode(operator='=', operands=[RelationNode(name='a'), Operand('10')])
            ),
            SelectionNode(operator='=', operands=[RelationNode(name='a'), Operand('10')]),
            True
        ),
        (
            (
                SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('10')]),
                SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('1')])
            ),
            SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('10')]),
            True
        ),
        (
            (
                SelectionNode(operator='>=', operands=[RelationNode(name='a'), Operand('1')]),
                SelectionNode(operator='<=', operands=[RelationNode(name='a'), Operand('1')])
            ),
            SelectionNode(operator='=', operands=[RelationNode(name='a'), Operand('1')]),
            True
        ),
        (
            (
                SelectionNode(operator='>', operands=[RelationNode(name='a'), Operand('1')]),
                SelectionNode(operator='<', operands=[RelationNode(name='a'), Operand('1')])
            ),
            SelectionNode(operator='<', operands=[RelationNode(name='a'), Operand('1')]),
            False
        ),

    ]

    for td in test_data:
        inputs, expected_b, expected_out = td
        a, b = inputs
        actual = merge_selections_(a, b)

        assert actual == expected_out
        assert str(expected_b) == str(b)


def test_tree_selection_push_down_other():
    query = """
        SELECT e.id, e.name 
        FROM employees as e, departments as d
        WHERE e.dept = d.id AND
            d.id < 10 AND
            e.id > 10
        """

    actual = parser.parse_sql(query)
    print(actual.pretty())

    expected = RelationalTree(
        sql=query,
        root=ProjectionNode.mock([
            JoinNode.mock([
                SelectionNode.mock([
                    RelationNode()
                ]),
                SelectionNode.mock([
                    RelationNode()
                ])
            ])
        ])
    )

    actual.optimize()
    assert_equal(actual, expected, eq_mode=EqualityMode.STRUCTURE)


def test_tree_selection_push_down_self():
    query = """
        SELECT m2.emp_name, m2.emp_salary, m2.man_name, m2.man_salary
        FROM (
            SELECT e.name as emp_name, e.salary as emp_salary, m.name as man_name, m.salary as man_salary
            FROM employees as e, 
                (select id, name, salary from employees where job ='managers' AND active=TRUE) as m
            WHERE e.manager_id = m.id
        ) as m2
        WHERE m2.emp_salary < 100 AND
            m2.man_salary > 100 
        """

    actual = parser.parse_sql(query)
    print(actual.pretty())

    expected = RelationalTree(
        sql=query,
        root=ProjectionNode.mock([
            AliasNode.mock([
                ProjectionNode.mock([
                    JoinNode.mock([
                        AliasNode.mock([
                            ProjectionNode.mock([
                                SelectionNode.mock([
                                    SelectionNode.mock([
                                        SelectionNode.mock([
                                            RelationNode()
                                        ])
                                    ])
                                ])
                            ]),
                        ]),
                        SelectionNode.mock([
                            RelationNode()
                        ]),
                    ])
                ])
            ])
        ])
    )

    actual.optimize()
    print(actual.pretty())
    assert_equal(actual, expected, eq_mode=EqualityMode.STRUCTURE)


def test_tree_selection_pushdown_merge():
    query = """
            SELECT e.id, e.name 
            FROM (SELECT id,name from employees where 10 < id) as e, 
                employees as b
            WHERE e.id = b.id AND
             e.id > 1
            """

    actual = parser.parse_sql(query)
    print(actual.pretty())

    expected = RelationalTree(
        sql=query,
        root=ProjectionNode.mock([
            JoinNode.mock([
                AliasNode.mock([
                    ProjectionNode.mock([
                        SelectionNode.mock([
                            RelationNode()
                        ])
                    ])
                ]),
                RelationNode()
            ])
        ])
    )

    actual.optimize()
    print(actual.pretty())
    assert_equal(actual, expected, eq_mode=EqualityMode.STRUCTURE)
