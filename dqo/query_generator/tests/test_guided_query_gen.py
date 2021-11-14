import mock
import numpy as np
import random

from dqo.db.tests.datasets import employees_db_w_meta
from dqo.db.tests.fake_client import FakeClient
from dqo.query_generator.guided import GuidedQueryGen
from relational import SQLParser

fake_db = FakeClient(
    db=employees_db_w_meta()
)


def test_smoke():
    qn = GuidedQueryGen(fake_db, target=(2 ** 7, 2 ** 8), max_steps=1)
    qn.randomize_initial()
    qn.guide()

    assert qn.steps == 1


def test_initial():
    np.random.seed(0)
    random.seed(0)

    qn = GuidedQueryGen(fake_db, target=(2 ** 7, 2 ** 8))
    qn.randomize_initial()

    assert qn.current_sql == 'SELECT MIN(companies.id), MIN(companies.name) FROM companies WHERE companies.id != 1484405'


def test_step_broaden_valid():
    np.random.seed(0)
    random.seed(0)

    qn = GuidedQueryGen(fake_db, target=(2 ** 7, 2 ** 8), seed=0)
    qn.qb.q = SQLParser.to_query('SELECT MIN(employees.id) FROM employees')
    qn.qb.sync()

    assert len(list(qn.qb.q.relations)) == 1
    assert len(list(qn.qb.q.projections)) == 1

    qn.step = mock.MagicMock(wraps=qn.step)
    fake_runtime = 2 ** 6
    qn.run_query = mock.MagicMock(return_value=(fake_runtime, False))

    runtime, action_name, done = qn.step(2 ** 4)

    assert fake_runtime == runtime
    assert action_name == qn.broaden.__name__
    assert done is False

    assert qn.current_sql == 'SELECT MIN(employees.active), MIN(employees.id) FROM employees'


def test_step_broaden_invalid():
    np.random.seed(0)
    random.seed(0)

    qn = GuidedQueryGen(fake_db, target=(2 ** 7, 2 ** 8), seed=0)
    qn.qb.q = SQLParser.to_query('SELECT MIN(employees.id) FROM employees')
    qn.qb.sync()

    assert len(list(qn.qb.q.relations)) == 1
    assert len(list(qn.qb.q.projections)) == 1

    qn.step = mock.MagicMock(wraps=qn.step)
    fake_runtime = 2 ** 6
    qn.run_query = mock.MagicMock(return_value=(fake_runtime, False))

    runtime, action_name, done = qn.step(2 ** 4)

    assert fake_runtime == runtime
    assert action_name == qn.broaden.__name__
    assert done is False

    assert qn.current_sql == 'SELECT MIN(employees.active), MIN(employees.id) FROM employees'


def test_step_narrow_valid():
    np.random.seed(1)
    random.seed(1)

    qn = GuidedQueryGen(fake_db, target=(2 ** 5, 2 ** 6), seed=0)
    qn.qb.q = SQLParser.to_query('SELECT MIN(employees.id), MIN(departments.id) FROM employees, departments WHERE employees.dept = departments.id')
    qn.qb.sync()

    assert len(list(qn.qb.q.relations)) == 2
    assert len(list(qn.qb.q.projections)) == 2

    qn.step = mock.MagicMock(wraps=qn.step)
    fake_runtime = 2 ** 5
    qn.run_query = mock.MagicMock(return_value=(fake_runtime, False))

    runtime, action_name, done = qn.step(2 ** 7)

    assert fake_runtime == runtime
    assert action_name == qn.narrow.__name__
    assert done is False

    assert qn.current_sql == 'SELECT MIN(departments.id) FROM departments'


def test_step_narrow_invalid():
    np.random.seed(0)
    random.seed(0)

    qn = GuidedQueryGen(fake_db, target=(2 ** 5, 2 ** 6), seed=0)
    qn.qb.q = SQLParser.to_query('SELECT MIN(employees.id) FROM employees')
    qn.qb.sync()

    assert len(list(qn.qb.q.relations)) == 1
    assert len(list(qn.qb.q.projections)) == 1

    qn.step = mock.MagicMock(wraps=qn.step)
    fake_runtime = 2 ** 5
    qn.run_query = mock.MagicMock(return_value=(fake_runtime, False))

    runtime, action_name, done = qn.step(2 ** 7)

    assert fake_runtime == runtime
    assert action_name == qn.narrow.__name__
    assert done is False

    assert qn.current_sql == "SELECT MIN(employees.id) FROM employees WHERE employees.name LIKE '%vtk%'"
