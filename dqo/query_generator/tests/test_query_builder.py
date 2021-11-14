import numpy as np

from dqo.db.models import Database, Table, Column, DataType, ColumnMeta
from dqo.db.tests.datasets import employees_db_w_meta
from dqo.query_generator.query_builder import QueryBuilder


def test_basic_actions():
    np.random.seed(0)

    qb = QueryBuilder(schema=employees_db_w_meta())
    assert len(qb.q) == 0

    qb.add_projection()
    qb.add_condition()
    assert len(qb.q) == 0

    qb.add_relation()
    assert len(list(qb.q.relations)) == 1
    assert len(list(qb.q.projections)) == 1

    qb.add_condition()
    assert len(list(qb.q.conditions)) == 1

    qb.add_projection()
    assert len(list(qb.q.projections)) == 2

    sql = qb.q.to_sql(alias=False, pretty=False)
    assert len(sql) > 5


def test_remove_cascade_single_table():
    np.random.seed(0)

    qb = QueryBuilder(schema=employees_db_w_meta())

    qb.add_relation()
    qb.add_condition()

    qb.remove_projection()
    assert len(list(qb.q.relations)) == 0
    assert len(list(qb.q.projections)) == 0

    qb.add_relation()
    assert len(list(qb.q.relations)) == 1
    assert len(list(qb.q.projections)) == 1


def test_remove_relation_cascade_single_table():
    np.random.seed(0)

    qb = QueryBuilder(schema=employees_db_w_meta())

    qb.add_relation()
    qb.add_projection()
    qb.add_condition()

    qb.remove_relation()

    assert len(list(qb.q.relations)) == 0
    assert len(list(qb.q.projections)) == 0
    assert len(list(qb.q.relations)) == 0


def test_remove_cascade_multi_table():
    np.random.seed(0)

    qb = QueryBuilder(schema=employees_db_w_meta())

    qb.add_relation()
    qb.add_relation()

    assert len(list(qb.q.relations)) == 2
    assert len(list(qb.q.selections)) == 1

    qb.remove_relation()
    assert len(list(qb.q.relations)) == 1
    assert len(list(qb.q.selections)) == 0


def test_add_more_conditions_than_columns():
    np.random.seed(0)

    qb = QueryBuilder(
        schema=Database(
            tables=[
                Table("employees", [
                    Column("id", DataType.STRING, meta=ColumnMeta(int(1e6), 0, int(1e6), True)),
                    Column("salary", DataType.NUMBER, meta=ColumnMeta(int(1e6), 10, int(1e5))),
                ])
            ]
        )
    )

    qb.add_relation()
    for _ in range(3):
        qb.add_condition()

    assert len(qb.q._relations) == 1
    assert len(qb.q._conditions) == 3


def test_add_all_projections():
    np.random.seed(0)

    qb = QueryBuilder(
        schema=Database(
            tables=[
                Table("employees", [
                    Column("id", DataType.STRING, meta=ColumnMeta(int(1e6), 0, int(1e6), True)),
                    Column("salary", DataType.NUMBER, meta=ColumnMeta(int(1e6), 10, int(1e5))),
                ])
            ]
        )
    )

    qb.add_relation()
    for _ in range(3):
        qb.add_projection()

    assert len(qb.q._relations) == 1
    assert len(qb.q._projections) == 2
