from dqo.db.models import Table, Column, DataType, Database, ColumnStats, NumericStats


def employees_db_w_meta() -> Database:
    table_employees = Table("employees",
                            [
                                Column("id", DataType.STRING, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                                Column("salary", DataType.NUMBER, stats=ColumnStats(int(1e6), 10, int(1e5))),
                                Column("dept", DataType.STRING, stats=ColumnStats(int(1e6), 100, 100)),
                                Column("company", DataType.STRING, stats=ColumnStats(int(1e6), 0, 3)),
                                Column("name", DataType.STRING, stats=ColumnStats(int(1e6), 0, int(1e5))),
                                Column("active", DataType.BOOL, stats=ColumnStats(int(1e6), 0, 2))
                            ])
    table_departments = Table("departments",
                              [
                                  Column("id", DataType.NUMBER, stats=ColumnStats(100, 0, 100, True)),
                                  Column("name", DataType.STRING, stats=ColumnStats(100, 0, 100))
                              ])
    table_companies = Table("companies",
                            [
                                Column("id", DataType.NUMBER, stats=ColumnStats(3, 0, 3, True)),
                                Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3))
                            ])
    return Database([
        table_employees,
        table_departments,
        table_companies
    ])


def employees2_db_w_meta() -> Database:
    table_employees = Table("employees",
                            [
                                Column("id", DataType.STRING, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                                Column("salary", DataType.FLOAT,
                                       stats=ColumnStats(
                                           total=int(1e6), nulls=10, distinct=int(1e5),
                                           values=NumericStats(min=1.0, max=100.0, mean=50, variance=0.1, skewness=0.1, kurtosis=0.1, freq=[],
                                                               hist=[])
                                       )),
                                Column("date", DataType.TIME,
                                       stats=ColumnStats(
                                           total=int(1e6), nulls=10, distinct=int(1e5),
                                           values=NumericStats(
                                               min=1279568347, max=1410661888, mean=50, variance=0.1, skewness=0.1, kurtosis=0.1,
                                               freq=[],
                                               hist=[])
                                       )),

                            ])
    table_departments = Table("departments",
                              [
                                  Column("id", DataType.NUMBER, stats=ColumnStats(100, 0, 100, True)),
                                  Column("name", DataType.STRING, stats=ColumnStats(100, 0, 100))
                              ])
    table_companies = Table("companies",
                            [
                                Column("id", DataType.NUMBER, stats=ColumnStats(3, 0, 3, True)),
                                Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3))
                            ])
    return Database([
        table_employees,
        table_departments,
        table_companies
    ])


def imdb_db_w_meta() -> Database:
    aka_name = Table("aka_name",
                     [
                         Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                         Column("person_id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                         Column("imdb_index", DataType.STRING, stats=ColumnStats(int(1e6), 10, int(1e5))),
                         Column("surname_pcode", DataType.STRING, stats=ColumnStats(int(1e6), 0, int(1e5)))
                     ])
    comp_cast_type = Table("comp_cast_type",
                           [
                               Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                               Column("name", DataType.STRING, stats=ColumnStats(100, 0, 100))
                           ])
    company_name = Table("company_name",
                         [
                             Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                             Column("md5sum", DataType.STRING, stats=ColumnStats(3, 0, 3)),
                             Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3))
                         ])

    info_type = Table("info_type",
                      [
                          Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                          Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3)),
                          Column("info", DataType.STRING, stats=ColumnStats(3, 0, 3)),

                      ])

    keyword = Table("keyword",
                    [
                        Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                        Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3)),
                        Column("phonetic_code", DataType.STRING, stats=ColumnStats(3, 0, 3))

                    ])

    movie_info = Table("movie_info",
                       [
                           Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                           Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3)),
                           Column("movie_id", DataType.NUMBER, stats=ColumnStats(3, 0, 3))
                       ])

    movie_info_idx = Table("movie_info_idx",
                           [
                               Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                               Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3)),
                               Column("info_type_id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),

                           ])

    movie_keyword = Table("movie_keyword",
                          [
                              Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                              Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3)),
                              Column("movie_id", DataType.NUMBER, stats=ColumnStats(3, 0, 3))
                          ])

    movie_link = Table("movie_link",
                       [
                           Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                           Column("linked_movie_id", DataType.NUMBER, stats=ColumnStats(3, 0, 3)),
                           Column("movie_id", DataType.NUMBER, stats=ColumnStats(3, 0, 3))
                       ])

    role_type = Table("role_type",
                      [
                          Column("id", DataType.NUMBER, stats=ColumnStats(int(1e6), 0, int(1e6), True)),
                          Column("name", DataType.STRING, stats=ColumnStats(3, 0, 3))
                      ])
    return Database([
        aka_name,
        comp_cast_type,
        company_name,
        movie_info,
        movie_info_idx,
        movie_keyword,
        movie_link,
        role_type,
        info_type,
        keyword
    ])
