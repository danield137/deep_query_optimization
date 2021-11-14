from dqo.db.tests.datasets import employees_db_w_meta
from dqo.relational import SQLParser
from dqo.relational.query import Query

test_db = employees_db_w_meta()


def test_well_defined_simple():
    q = Query()

    q.add_join([
        test_db['employees']['dept'].to_ref(),
        test_db['departments']['id'].to_ref()
    ])
    q.add_projection(col=test_db['employees']['salary'].to_ref())
    q.add_selection(test_db['employees']['active'].to_ref(), '=', 'TRUE')

    expected = "SELECT t2.salary \n" + \
               "FROM departments as t1, employees as t2 \n" + \
               "WHERE t2.active = TRUE AND \n" \
               "      t2.dept = t1.id"

    actual = q.to_sql()

    assert actual == expected


def test_well_defined_simple_order():
    q = Query(track_order=True)

    q.add_join([
        test_db['employees']['dept'].to_ref(),
        test_db['departments']['id'].to_ref()
    ], with_relations=True)
    q.add_projection(col=test_db['employees']['salary'].to_ref())
    q.add_selection(test_db['employees']['active'].to_ref(), '=', 'TRUE')

    expected = "SELECT t1.salary \n" + \
               "FROM employees as t1, departments as t2 \n" + \
               "WHERE t1.dept = t2.id AND \n" \
               "      t1.active = TRUE"

    actual = q.to_sql()

    assert actual == expected


def test_projections_only():
    q = Query()

    q.add_projection(test_db['employees']['salary'].to_ref(), 'MIN')

    expected = "SELECT MIN(t1.salary) \n" + \
               "FROM employees as t1"

    actual = q.to_sql()
    assert actual == expected


def test_projections_and_selections_only():
    q = Query()

    q.add_projection(test_db['employees']['salary'].to_ref())
    q.add_selection(test_db['employees']['salary'].to_ref(), '>', 30)

    expected = "SELECT t1.salary \n" + \
               "FROM employees as t1 \n" + \
               "WHERE t1.salary > 30"

    actual = q.to_sql()
    assert actual == expected


def test_sql_to_query_and_back_w_join():
    sql = "SELECT e.id, e.salary FROM departments as d, employees as e WHERE e.salary > 1000 AND e.dept = d.id"
    q = SQLParser.to_query(sql)

    assert len(list(q.projections)) == 2
    assert len(list(q.relations)) == 2
    assert len(list(q.conditions)) == 1
    assert len(list(q.joins)) == 1
    assert len(list(q.selections)) == 2

    actual = q.to_sql(pretty=False)

    assert actual == sql


def test_sql_to_query_and_back():
    sql = "SELECT e.id, e.salary FROM employees as e WHERE e.salary > 1000"
    q = SQLParser.to_query(sql)

    assert len(list(q.projections)) == 2
    assert len(list(q.relations)) == 1
    assert len(list(q.conditions)) == 1

    actual = q.to_sql(pretty=False)

    assert actual == sql


