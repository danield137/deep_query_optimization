from dqo.db.tests.datasets import employees_db_w_meta
from dqo.relational import SQLParser
from dqo.relational import parse_tree

test_db = employees_db_w_meta()


def test_condition_permutation():
    sql = """
       SELECT MIN(employees.salary) 
           FROM employees
           WHERE employees.id > 200
       """

    rel_tree = SQLParser.to_relational_tree(sql)

    permutations = rel_tree.permutations()

    assert len(permutations) == 2

    queries = [parse_tree(p, keep_order=True).to_sql(pretty=False, alias=False) for p in permutations]
    # ensure all are different textually
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            assert queries[i] != queries[j]

    # ensure they are all semantically the same
    sentry = permutations[0]
    for p in permutations[1:]:
        assert len(list(sentry.get_selections())) == len(list(p.get_selections()))
        assert len(list(sentry.get_projections())) == len(list(p.get_projections()))
        assert len(list(sentry.relations.keys())) == len(list(p.relations.keys()))


def test_join_permutation():
    sql = """
    SELECT MIN(employees.salary) 
        FROM employees, departments, companies
        WHERE employees.id = departments.id AND companies.id = departments.id
    """

    rel_tree = SQLParser.to_relational_tree(sql)

    permutations = rel_tree.permutations()

    assert len(permutations) == 4

    queries = [parse_tree(p, keep_order=True).to_sql(pretty=False, alias=False) for p in permutations]
    # ensure all are different textually
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            assert queries[i] != queries[j]

    # ensure they are all semantically the same
    sentry = permutations[0]
    for p in permutations[1:]:
        assert len(list(sentry.get_selections())) == len(list(p.get_selections()))
        assert len(list(sentry.get_projections())) == len(list(p.get_projections()))
        assert len(list(sentry.relations.keys())) == len(list(p.relations.keys()))


def test_conditions_permutation():
    sql = """
    SELECT MIN(employees.salary) 
        FROM employees
        WHERE employees.id > 1 AND employees.salary > 100 AND employees.salary < 200
    """

    rel_tree = SQLParser.to_relational_tree(sql)

    permutations = rel_tree.permutations()

    # assert len(permutations) == 6

    queries = [parse_tree(p, keep_order=True).to_sql(pretty=False, alias=False) for p in permutations]

    # ensure all are different textually
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            assert queries[i] != queries[j]

    # ensure they are all semantically the same
    sentry = permutations[0]
    for p in permutations[1:]:
        assert len(list(sentry.get_selections())) == len(list(p.get_selections()))
        assert len(list(sentry.get_projections())) == len(list(p.get_projections()))
        assert len(list(sentry.relations.keys())) == len(list(p.relations.keys()))


def test_join_and_selection_permutations():
    sql = """
        SELECT MIN(employees.salary) 
            FROM employees, departments
            WHERE employees.id > 1 AND employees.dept_id = departments.id 
        """

    rel_tree = SQLParser.to_relational_tree(sql)

    permutations = rel_tree.permutations()

    # assert len(permutations) == 8

    queries = [parse_tree(p, keep_order=True).to_sql(pretty=False, alias=False) for p in permutations]

    # ensure all are different textually
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            assert queries[i] != queries[j]

    # ensure they are all semantically the same
    sentry = permutations[0]
    for p in permutations[1:]:
        assert len(list(sentry.get_selections())) == len(list(p.get_selections()))
        assert len(list(sentry.get_projections())) == len(list(p.get_projections()))
        assert len(list(sentry.relations.keys())) == len(list(p.relations.keys()))
