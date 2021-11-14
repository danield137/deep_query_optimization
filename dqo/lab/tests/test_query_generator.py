import random

import pytest

from dqo.db.tests.datasets import employees_db_w_meta, employees2_db_w_meta
from dqo.relational import SQLParser
from dqo.relational.sql.bindings import validate
from dqo.query_generator import RandomQueryGen


@pytest.mark.order1
def test_emps():
    db = employees_db_w_meta()

    expected = "SELECT MIN(t1.active), MIN(t1.company), MIN(t1.dept), MIN(t1.id), MIN(t1.salary) \nFROM employees as t1 \nWHERE t1.company LIKE '%ubd%'"
    query = RandomQueryGen(db, seed=1).randomize().to_sql()

    rel_tree = SQLParser.to_relational_tree(query)

    assert validate(rel_tree, db) is None

    assert query == expected, query


@pytest.mark.skip('find a better way to test this, becase sql parser isnt good with functions')
def test_emps_2():
    db = employees2_db_w_meta()

    random.seed(13)
    query = RandomQueryGen(db).randomize().to_sql()

    rel_tree = SQLParser.to_relational_tree(query)

    assert validate(rel_tree, db) is None
    assert query == 'SELECT MIN(t2.id) \nFROM companies as t1, employees as t2 \nWHERE t1.name LIKE \'%smn%\' AND t2.salary < 75.32897343563423 AND t2.salary = 61.658577399704534 AND t1.name = t2.id'
