import pytest

from dqo.relational.sql.ast import AbstractSyntaxTree
from dqo.relational.tree import parser
from dqo.relational.tree.node import ProjectionNode


@pytest.mark.skip('handle astrix properly')
def test_parser_astrix():
    query = 'Select * from employees as e'
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 0
    assert len(result.get_selection_columns()) == 0
    assert len(result.get_projections()) == 1
    assert len(list(result.nodes())) == 2
    assert len(result.relations) == 1
    assert isinstance(result.root, ProjectionNode)


def test_parser_simple():
    query = 'Select e.id from employees as e'
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 0
    assert len(result.get_selection_columns()) == 0
    assert len(result.get_projections()) == 1
    assert len(list(result.nodes())) == 2
    assert len(result.relations) == 1
    assert isinstance(result.root, ProjectionNode)


@pytest.mark.skip("not implemented")
def test_parser_simple_subselect():
    query = 'Select e.id, b.id from employees e, (SELECT id from employees where salary >= 100) as b'
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 0
    assert len(result.get_selection_columns()) == 0
    assert len(result.get_projections()) == 1
    assert len(result.relations) == 1
    assert isinstance(result.root, ProjectionNode)


def test_parser_simple_col_gt_number():
    query = """SELECT e.id 
               FROM employees as e 
               WHERE e.salary > 100"""
    ast = AbstractSyntaxTree.parse(query)

    result = parser.parse_ast(ast.root, ast.query)
    print(result.pretty())

    assert len(result.get_joins()) == 0
    assert len(result.get_selection_columns()) == 1
    assert len(result.get_projections()) == 1

    assert len(result.relations) == 1
    assert isinstance(result.root, ProjectionNode)


def test_parser_simple_same_col_twice():
    query = """SELECT e.id, e.salary 
               FROM employees as e 
               WHERE e.salary > 100 and e.salary < 200"""
    ast = AbstractSyntaxTree.parse(query)

    result = parser.parse_ast(ast.root, ast.query)
    print(result.pretty())

    assert len(result.get_joins()) == 0
    assert len(result.get_selection_columns()) == 1
    assert len(result.get_projections()) == 1
    assert len(result.get_projections()[0].columns) == 2
    assert len(result.relations) == 1
    assert isinstance(result.root, ProjectionNode)


def test_parser_simple_same_relation_same_col_twice():
    query = 'Select e1.id, e2.id ' \
            'from employees as e1, employees as e2 ' \
            'where e1.salary > 100 and e2.salary < 200'
    ast = AbstractSyntaxTree.parse(query)

    result = parser.parse_ast(ast.root, ast.query)
    print(result.pretty())

    assert len(result.get_joins()) == 0
    # TODO: not sure this should actually count as 2
    assert len(result.get_selection_columns()) == 2
    assert len(result.get_projections()) == 1
    assert len(result.get_projections()[0].columns) == 2
    assert len(result.relations) == 2
    assert isinstance(result.root, ProjectionNode)


def test_parser_simple_two_tables_lte_and_join():
    query = """Select e.id as eid, d.id as did 
    From employees as e , departments as d 
    where e.salary <= 100 AND d.id = e.dept"""
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 1
    assert len(result.get_selection_columns()) == 3
    assert len(result.relations) == 2
    assert len(result.get_projections()[0].columns) == 2
    assert isinstance(result.root, ProjectionNode)


def test_parser_simple_two_tables_w_aliases_join_and_neq():
    query = """
    SELECT e.id as eid, d.id as did 
        FROM employees as e , departments as d 
        WHERE d.id != 100 AND d.id = e.dept
    """
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 1
    assert len(result.get_selection_columns()) == 2
    assert len(result.relations) == 2
    assert len(result.get_projections()[0].columns) == 2
    assert isinstance(result.root, ProjectionNode)


def test_parser_simple_not_like_and_true():
    query = """
    Select e.id as eid
    From employees as e 
    where e.active IS NOT FALSE 
        AND e.name NOT LIKE '%DEAD%' 
        AND e.job = 'manager' 
    """
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 0
    assert len(result.get_selection_columns()) == 3
    assert len(result.relations) == 1
    assert len(result.get_projections()[0].columns) == 1
    assert isinstance(result.root, ProjectionNode)


def test_parser_advance_subselect_with_eq_and_joins():
    query = """
    Select e.id as eid, d.id as did, managers.rewards
    From employees as e, 
         departments as d,
         (select id,rewards from employees where job='manager') as managers
    where e.salary > 100 AND 
          d.id = e.dept AND 
          managers.id = e.manager_id AND 
          managers.rewards = 1 
    """
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 2
    assert len(result.get_selection_columns()) == 7
    assert len(result.get_projections()) == 2
    assert len(result.get_projections()[0].columns) == 3
    assert len(result.relations) == 3
    assert isinstance(result.root, ProjectionNode)


def test_parser_advance_and_or():
    query = """
    Select e.id as eid, d.id as did
    From employees as e ,
         departments as d
    where (e.salary > 100 OR d.active = TRUE) AND 
         d.id = e.dept
    """
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 1
    assert len(result.get_selection_columns()) == 4
    assert len(result.relations) == 2
    assert len(result.get_projections()) == 1
    assert len(result.get_projections()[0].columns) == 2
    assert isinstance(result.root, ProjectionNode)


def test_parser_multiple_or_nodes():
    query = """
    Select e.id as eid, d.id as did
    From employees as e ,
         departments as d
    where (e.salary > 100 OR d.active = TRUE OR e.salary < 10) AND 
         d.id = e.dept
    """
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 1
    assert len(result.get_selection_columns()) == 4
    assert len(result.relations) == 2
    assert len(result.get_projections()) == 1
    assert len(result.get_projections()[0].columns) == 2
    assert isinstance(result.root, ProjectionNode)
