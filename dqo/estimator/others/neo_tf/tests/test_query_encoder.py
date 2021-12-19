from dqo.db.models import Database, Table, Column, DataType
from estimator.others.neo.v1.encoder import encode_query


# TODO: should be a real function


def get_example_db_schema():
    return Database([
        Table("A",
              [
                  Column("COL_1", DataType.NUMBER),
                  Column("COL_2", DataType.NUMBER),
                  Column("COL_3", DataType.NUMBER),
                  Column("COL_4", DataType.NUMBER),
                  Column("COL_5", DataType.NUMBER)
              ]),
        Table("B",
              [
                  Column("COL_1", DataType.NUMBER),
                  Column("COL_2", DataType.NUMBER),
                  Column("COL_3", DataType.NUMBER),
                  Column("COL_4", DataType.NUMBER),
                  Column("COL_5", DataType.NUMBER)
              ]),
        Table("C",
              [
                  Column("COL_1", DataType.NUMBER),
                  Column("COL_2", DataType.NUMBER),
                  Column("COL_3", DataType.NUMBER),
                  Column("COL_4", DataType.NUMBER),
                  Column("COL_5", DataType.NUMBER)
              ]),
        Table("D",
              [
                  Column("COL_1", DataType.NUMBER),
                  Column("COL_2", DataType.NUMBER),
                  Column("COL_3", DataType.NUMBER),
                  Column("COL_4", DataType.NUMBER),
                  Column("COL_5", DataType.NUMBER)
              ]),
        Table("E",
              [
                  Column("COL_1", DataType.NUMBER),
                  Column("COL_2", DataType.NUMBER),
                  Column("COL_3", DataType.NUMBER),
                  Column("COL_4", DataType.NUMBER),
                  Column("COL_5", DataType.NUMBER)
              ])])


def test_encoder_no_aliases():
    example = """
    SELECT A.COL_3, C.COL_3 
    FROM A, B, C, D
    WHERE A.COL_3 = C.COL_3 
        AND A.COL_4 = D.COL_4 
        AND C.COL_5 = B.COL_5
        AND A.COL_2 > 5 AND B.COL_1 = 'h'

    """

    db = get_example_db_schema()
    query_encoding = encode_query(db, example)
    assert query_encoding == [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_encoder_aliases():
    example = """
    SELECT a.COL_3  
    FROM A as a, B as b, C as c, D as d
    WHERE a.COL_3 = c.COL_3 
        AND a.COL_4 = d.COL_4 
        AND c.COL_5 = b.COL_5
        AND a.COL_2 > 5 AND b.COL_1 = 'h'

    """
    db = get_example_db_schema()
    #query_encoding = encode_query(db, example)
    #assert query_encoding == [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_encoder_duplicates():
    example1 = """
    SELECT a.COL_1
     FROM A as a, A as b
     WHERE a.COL_1 = b.COL_1 AND 
        a.COL_2 > 5 AND 
        b.COL_3 = 'h'

    """

    example2 = """
    SELECT a.COL_3 
     FROM A as a
     WHERE a.COL_1 = 1 AND 
        a.COL_2 > 5 AND 
        a.COL_3 = 'h'

    """
    db = get_example_db_schema()
    query_encoding1 = encode_query(db, example1)
    query_encoding2 = encode_query(db, example2)
    assert query_encoding1 == query_encoding2


# TODO: fix parser with astrix
def test_encoder_no_joins():
    example = """
    SELECT A.COL_2 
        FROM A, B, C, D
        WHERE A.COL_2 > 5 
            AND B.COL_1 = 'h' 
            and C.COL_3 > 1 
            and D.COL_1 = 2
    """
    db = get_example_db_schema()
    query_encoding = encode_query(db, example)
    assert query_encoding == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def test_encoder_all_joins():
    example = """
    SELECT MIN(A.COL_3), MAX(B.COL_4) 
    FROM A, B, C, D, E
    WHERE A.COL_3 = C.COL_3 
        AND A.COL_4 = D.COL_4 
        AND C.COL_5 = B.COL_5
        AND E.COL_4 = B.COL_4
        AND A.COL_2 > 5 
        AND B.COL_1 = 'h'

    """
    db = get_example_db_schema()
    query_encoding = encode_query(db, example)
    assert query_encoding == [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
