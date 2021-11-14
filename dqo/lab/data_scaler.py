from dqo.db.clients import DatabaseClient
from dqo.db.models import Table, DataType, Column, Database


def make_value(column: Column):
    return None


def make_row(table: Table) -> list:
    row = []
    for col in table.columns:
        row.append(make_value(col))


def inflate():
    pass


def deflate():
    pass


def scale(db_client: DatabaseClient, factor=2):
    db = db_client.model()

    for table in db.tables:
        target_count = table.stats.rows * factor

        if factor > 1:
            infalte()
        elif factor < 1:
            deflate()
