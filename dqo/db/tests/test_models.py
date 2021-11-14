from __future__ import annotations

from dqo.db.models import DataType, Column, Table, Database


def test_serialize_column():
    expected = Column('moshe', data_type=DataType.STRING)

    actual = Column.from_json(expected.to_json())
    assert actual.name == expected.name


def test_serialize_table():
    expected_column = Column('moshe', data_type=DataType.STRING)
    expected_table = Table('t1', [expected_column])

    actual_table = Table.from_json(expected_table.to_json())
    assert expected_table.name == actual_table.name
    assert expected_table.columns[0].name == actual_table.columns[0].name


def test_serialize_database():
    test_db = Database(
        tables=[
            Table(
                'employees',
                columns=[
                    Column('id', DataType.STRING),
                    Column('name', DataType.STRING),
                    Column('dept_id', DataType.STRING),
                    Column('salary', DataType.NUMBER),
                    Column('city_id', DataType.STRING),
                    Column('active', DataType.BOOL),
                ]
            ),
            Table(
                'departments',
                columns=[
                    Column('id', DataType.STRING),
                    Column('name', DataType.STRING),
                    Column('city_id', DataType.STRING)
                ]
            ),
            Table(
                'cities',
                columns=[
                    Column('id', DataType.STRING),
                    Column('name', DataType.STRING),
                    Column('population', DataType.NUMBER),
                ]
            ),
        ]
    )

    test_db_actual: Database = Database.from_json(test_db.to_json())

    assert test_db_actual.columns_count == test_db.columns_count

    assert test_db_actual.tables_lookup.keys() == test_db.tables_lookup.keys()

    for table in test_db.tables:
        assert repr(test_db_actual[table.name].columns) == repr(table.columns)
        assert test_db_actual[table.name].types_lookup.keys() == table.types_lookup.keys()
