import logging
import os
import re
import tempfile
from typing import List, Any, Tuple, cast, Dict
from urllib.parse import urlparse

import psycopg2.errorcodes
import psycopg2.extras
import sqlparse
from psycopg2 import connect
from psycopg2._psycopg import OperationalError
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from db.execution_plan import ExecutionPlan
from dqo.db.clients.base import DatabaseClient
from dqo.db.models import Column, Table, Database, DataType, NumericStats, StringStats, ColumnStats, TableStats

PARTITIONS_QUERY = """
select 
    par.relnamespace::regnamespace::text as schema, 
    par.relname as table_name, 
    partnatts as num_columns,
    column_index,
    col.column_name
from   
    (select
         partrelid,
         partnatts,
         case partstrat 
              when 'l' then 'list' 
              when 'r' then 'range' end as partition_strategy,
         unnest(partattrs) column_index
     from
         pg_partitioned_table) pt 
join   
    pg_class par 
on     
    par.oid = pt.partrelid
join
    information_schema.columns col
on  
    col.table_schema = par.relnamespace::regnamespace::text
    and col.table_name = par.relname
    and ordinal_position = pt.column_index;"""

logger = logging.getLogger('db.clients.postgres')


def to_known_data_type(pg_type: str) -> DataType:
    if pg_type in ['double', 'float', 'numeric', 'real', 'double precision']:
        return DataType.FLOAT
    if pg_type in ['integer', 'bigint']:
        return DataType.NUMBER
    if pg_type in ['text', 'character varying', 'character', 'char']:
        return DataType.STRING
    if pg_type in []:
        return DataType.BOOL
    if 'time' in pg_type or 'date' in pg_type or 'interval' in pg_type:
        return DataType.TIME

    raise ValueError(f'Unknown type {pg_type}')


class StatQueries:
    @staticmethod
    def table_columns(table_name: str) -> str:
        """
        :param table_name:
        :return: column_name, data_type, character_maximum_length
        """
        return """
        SELECT column_name,
               icols.data_type,
               icols.character_maximum_length
        FROM information_schema.columns icols
        WHERE table_name = '{table_name}'
        """.format(table_name=table_name)

    @staticmethod
    def table_sizes() -> str:
        """
        :param table_name:
        :return: table, pages
        """
        return """
SELECT relname                       as "table",
       greatest(relpages::bigint, 1) as pages
from pg_class
where relkind = 'r'
  and relnamespace = (select oid
                      from pg_namespace
                      where nspname = 'public');
"""

    @staticmethod
    def db_indexes() -> str:
        """
        :return: table_name, index_name, column_name
        """
        return """
        select t.relname as table_name,
               i.relname as index_name,
               a.attname as column_name
        from pg_class t,
             pg_class i,
             pg_index ix,
             pg_attribute a
        where t.oid = ix.indrelid
          and i.oid = ix.indexrelid
          and a.attrelid = t.oid
          and a.attnum = ANY (ix.indkey)
          and t.relkind = 'r'
          and t.relnamespace = (select oid
                              from pg_namespace
                              where nspname = 'public')
        order by t.relname,
                 i.relname;
        """

    @staticmethod
    def numeric_column_hist(column: str, table: str) -> str:
        """
        :param column:
        :param table:
        :return: bucket, (min,max), items
        """
        return """
        select width_bucket({column}, value_range.min, value_range.max + 1, 10) as bucket,
               min({column})::varchar || ',' || max({column})::varchar    as range,
               count(*)                                                     as freq
        from {table},
             (select min({column}) as min,
                     max({column}) as max
              from {table}) as value_range
        group by bucket
        order by bucket;
        """.format(column=column, table=table)

    @staticmethod
    def str_word_len_column_hist(column: str, table: str) -> str:
        """

        :param column:
        :param table:
        :return:
        """
        return """
select width_bucket(array_length(regexp_split_to_array({column}, E'\\s+'), 1), value_range.min, value_range.max + 1, 10) as bucket,
       min(array_length(regexp_split_to_array({column}, E'\\s+'), 1))::varchar || ',' || max(array_length(regexp_split_to_array({column}, E'\\s+'), 1))::varchar    as range,
       count(*)                                                     as freq
from {table},
     (select min(array_length(regexp_split_to_array({column}, E'\\s+'), 1)) as min,
             max(array_length(regexp_split_to_array({column}, E'\\s+'), 1)) as max
      from {table}) as value_range
group by bucket
order by bucket;
""".format(column=column, table=table)

    @staticmethod
    def str_len_column_hist(column: str, table: str) -> str:
        return """
            select width_bucket(length({column}), value_range.min, value_range.max + 1, 10) as bucket,
                   min(length({column}))::varchar || ',' || max(length({column}))::varchar    as range,
                   count(*)                                                     as freq
            from {table},
                 (select min(length({column})) as min,
                         max(length({column})) as max
                  from {table}) as value_range
            group by bucket
            order by bucket;
        """.format(column=column, table=table)

    @staticmethod
    def time_column_hist(column: str, table: str) -> str:
        return """
            select width_bucket(extract(epoch from {column}), value_range.min, value_range.max + 1, 10) as bucket,
                   min(extract(epoch from {column}))::varchar || ',' || max(extract(epoch from {column}))::varchar    as range,
                   count(*)                                                     as freq
            from {table},
                 (select min(extract(epoch from {column})) as min,
                         max(extract(epoch from {column})) as max
                  from {table}) as value_range
            group by bucket
            order by bucket;
        """.format(table=table, column=column)

    @staticmethod
    def table_columns_stats_query(table: Table) -> str:
        column_stats = []
        for col in table.columns:
            part = [
                f'count(*) FILTER ( WHERE {col.name} isnull ) as nulls_{col.name}',
                f'count(DISTINCT {col.name}) as distinct_{col.name}'
            ]

            if col.data_type in (DataType.NUMBER, DataType.FLOAT):
                part.append(f'stats_agg({col.name}) as stats_{col.name}')
            elif col.data_type == DataType.TIME:
                part.append(f'stats_agg(extract(epoch from {col.name})) as stats_{col.name}')
            elif col.data_type == DataType.STRING:
                part.append(f'stats_agg(length({col.name})) as length_stats_{col.name}')
                part.append(f"stats_agg(array_length(regexp_split_to_array({col.name}, E'\\s+'), 1)) as word_stats_{col.name}")

            column_stats.append(',\n'.join(part))

        cols_str = ',\n'.join(column_stats) + '\n'

        return f'SELECT count(*) as total, ' \
               f'{cols_str} ' \
               f'FROM {table.name}'

    @staticmethod
    def table_count(table: str) -> str:
        return """
        SELECT count(*) as total
        FROM {table}
        """.format(table=table)

    @staticmethod
    def column_stats_queries(table_name: str, column_name: str, data_type: DataType) -> List[str]:
        select_wrapper = """SELECT {clause} FROM {table_name}"""

        distinct_part = """count(*) FILTER ( WHERE {col} isnull ) as nulls_{col},
            count(DISTINCT {col}) as distinct_{col} \n""".format(col=column_name)
        parts = [
            distinct_part
        ]

        if data_type in (DataType.NUMBER, DataType.FLOAT):
            parts.append(f'stats_agg({column_name}) as stats_{column_name}')
        elif data_type == DataType.TIME:
            parts.append(f'stats_agg(extract(epoch from {column_name})) as stats_{column_name}')
        elif data_type == DataType.STRING:
            parts.append(f'stats_agg(length({column_name})) as length_stats_{column_name}')
            parts.append(f"stats_agg(array_length(regexp_split_to_array({column_name}, E'\\s+'), 1)) as word_stats_{column_name}")

        return [select_wrapper.format(clause=p, table_name=table_name) for p in parts]


class Postgres(DatabaseClient):
    ANALYZE_TIME_REGEX = re.compile(r"(?:[a-zA-Z ]*): (\d*.\d*) ms")

    def __init__(self, connection: str, query_timeout_secs: int = 600, readonly=True, **kwargs):
        uri = urlparse(connection)
        self.hostname = uri.hostname
        self.readonly = readonly
        self.db_name = uri.path.replace('/', '')
        self._kwargs = kwargs
        self._schema = None
        self.query_timeout_secs = query_timeout_secs

        try:
            self.conn = connect(host=self.hostname, dbname=self.db_name, **kwargs)
            if readonly:
                # this is to escape transaction errors - it's meaningless because we don't really insert data
                self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

            os.environ['PGOPTIONS'] = f'-c statement_timeout={query_timeout_secs}s'
            logger.info(f'set query limit to {query_timeout_secs}s')
        except OperationalError as e:
            if e.pgcode != psycopg2.errorcodes.QUERY_CANCELED:
                raise e

    def reset(self):
        self.conn = connect(host=self.hostname, dbname=self.db_name, **self._kwargs)
        if self.readonly:
            # this is to escape transaction errors - it's meaningless because we don't really insert data
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    def execute(self, query, as_dict=True, collect=True, log=True) -> List[Any]:
        if as_dict:
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        else:
            cur = self.conn.cursor()
        try:
            cur.execute(query)
            return cur.fetchall() if collect else []
        except OperationalError as e:
            if e.pgcode != psycopg2.errorcodes.QUERY_CANCELED:
                self.reset()
            if log:
                logger.info(f'query: {query} erred {str(e)}')
            try:
                cur.execute("ROLLBACK")
                self.conn.commit()
            except Exception as ex:
                logger.info(f'query: rollback erred {str(ex)}')
                self.reset()
            return []

    def time(self, query: str) -> float:
        results = self.execute(f'EXPLAIN ANALYZE ' + query)

        if not results:
            return -1.0

        planing_time_row, execution_time_row = results[-2:]

        execution_time = (float(self.ANALYZE_TIME_REGEX.match(planing_time_row[0])[1]) +
                          float(self.ANALYZE_TIME_REGEX.match(execution_time_row[0])[1]))

        return execution_time

    def analyze(self, query: str) -> Tuple[float, float, str]:
        results = self.execute(f'EXPLAIN (ANALYZE, FORMAT JSON )' + query)

        if not results:
            return -1.0, -1.0, ""
        json_results = results[0][0][0]

        exec_plan = json_results['Plan']
        planing_time = json_results['Planning Time']
        execution_time = json_results['Execution Time']

        return planing_time, execution_time, exec_plan

    def humanize_target(self):
        return f'{self.hostname}_{self.db_name}'

    def model(self, use_cache: bool = True) -> Database:
        # this is a hack to allow injecting a schema and reducing runtime
        if self._schema:
            return self._schema

        """
        path -- Base path to look for stored database schemes.
                Search pattern is '{path}/{hostname}/{db_name}.json
        """
        base_path = os.path.join(tempfile.gettempdir(), 'dqo', 'db', 'postgres')

        save_path = os.path.join(
            base_path,
            self.hostname,
            f'{self.db_name}.json'
        )

        if use_cache and os.path.exists(save_path):
            logger.info(f'Found existing file, loading from "{save_path}"')

            return Database.load(save_path)

        tables = []

        tables_stats = self.execute(StatQueries.table_sizes())
        logger.info(f'found {len(tables_stats)} tables.')
        for i, table_stats in enumerate(tables_stats):
            table_name = table_stats['table']

            columns = []
            column_stats = self.execute(StatQueries.table_columns(table_name))
            for result_column in column_stats:
                column_name = result_column['column_name']

                columns.append(Column(
                    name=column_name,
                    data_type=to_known_data_type(result_column['data_type'])
                ))

            table = Table(table_name, columns)
            self.update_table_stats_(table)
            self.update_histograms_(table)
            table.stats.pages = table_stats['pages']

            logger.info(f'({i + 1}/{len(tables_stats)}): finished reading {table.name} schema. Table has {table.stats.rows} rows.')
            tables.append(table)

        db = Database(tables)

        db_indexes = self.execute(StatQueries.db_indexes())
        for db_index in db_indexes:
            db[db_index['table_name']][db_index['column_name']].stats.index = True

        logger.info(f'saving results to "{save_path}"')
        db.save(save_path)

        return db

    @staticmethod
    def csvify(query):
        single_quote = '"'
        double_quote = '""'
        rn_char = '\r\n'
        n_char = '\n'
        whitespace = " "
        escaped_query = query.replace(single_quote, double_quote).replace(rn_char, whitespace).replace(n_char, whitespace)
        return f'"{escaped_query}"'

    def query_table_stats(self, table: Table) -> Dict[str, Any]:
        try:
            select = StatQueries.table_columns_stats_query(table)
            stats = dict(self.execute(select, log=False)[0])
        except:
            result = self.execute(StatQueries.table_count(table=table.name))
            stats = dict(result[0])
            logger.info(f'table stats for {table.name} failed as table probably too big ({stats.get("count")}). breaking up into column: ')

            for i, column in enumerate(table.columns):
                for select in StatQueries.column_stats_queries(table.name, column.name, column.data_type):
                    result = self.execute(select)
                    column_stats = dict(result[0])
                    stats.update(**column_stats)
                logger.info(f'{i + 1}/{len(table.columns)} column stats for {table.name}.{column.name} are ready.')
        return stats

    def update_table_stats_(self, table: Table):
        stats = self.query_table_stats(table)

        table.stats = TableStats(stats['total'], 1, 8 * 1024)  # 8KB is default

        for column in table.columns:
            # For now, time is treated as a number, because it is easier
            if column.data_type in (DataType.NUMBER, DataType.FLOAT, DataType.TIME):
                stats_tuple = stats.get(f'stats_{column.name}')[1:-1].split(',')
                value_stats = NumericStats(
                    min=float(stats_tuple[1] or 0),
                    max=float(stats_tuple[2] or 0),
                    mean=float(stats_tuple[3] or 0),
                    variance=float(stats_tuple[4] or 0),
                    skewness=float(stats_tuple[5] or 0),
                    kurtosis=float(stats_tuple[6] or 0),
                    hist=[],  # will be set later
                    freq=[]  # will be set later
                )
            elif column.data_type == DataType.STRING:
                length_stats_tuple = stats.get(f'length_stats_{column.name}')[1:-1].split(',')
                word_stats_tuple = stats.get(f'word_stats_{column.name}')[1:-1].split(',')
                value_stats = StringStats(
                    length=NumericStats(
                        min=float(length_stats_tuple[1] or 0),
                        max=float(length_stats_tuple[2] or 0),
                        mean=float(length_stats_tuple[3] or 0),
                        variance=float(length_stats_tuple[4] or 0),
                        skewness=float(length_stats_tuple[5] or 0),
                        kurtosis=float(length_stats_tuple[6] or 0),
                        hist=[],  # will be set later
                        freq=[]  # will be set later
                    ), word=NumericStats(
                        min=float(word_stats_tuple[1] or 0),
                        max=float(word_stats_tuple[2] or 0),
                        mean=float(word_stats_tuple[3] or 0),
                        variance=float(word_stats_tuple[4] or 0),
                        skewness=float(word_stats_tuple[5] or 0),
                        kurtosis=float(word_stats_tuple[6] or 0),
                        hist=[],  # will be set later
                        freq=[]  # will be set later
                    )
                )
            else:
                raise NotImplementedError()

            table[column.name].stats = ColumnStats(
                index=False,  # will be set later
                values=value_stats,
                total=stats.get('total') or 0,
                nulls=stats.get(f'nulls_{column.name}') or 0,
                distinct=stats.get(f'distinct_{column.name}') or 0
            )

    def update_histograms_(self, table: Table):
        def results_to_hist_freq(results) -> Tuple[List[float], List[float]]:
            if len(results) == 1:
                return [], []

            h, f = [], []
            for r in results:
                if r[0] is not None:
                    h.append(eval(r[1])[1])
                else:
                    h.append(None)
                f.append(int(r[2]))

            return h, f

        for column in table.columns:
            # For now, time is treated as a number, because it is easier
            if column.data_type in (DataType.NUMBER, DataType.FLOAT):
                column.stats.values = cast(NumericStats, column.stats.values)
                hist, freq = results_to_hist_freq(
                    self.execute(StatQueries.numeric_column_hist(column=column.name, table=table.name))
                )

                column.stats.values.hist = hist
                column.stats.values.freq = freq

            elif column.data_type == DataType.TIME:
                column.stats.values = cast(NumericStats, column.stats.values)
                hist, freq = results_to_hist_freq(
                    self.execute(StatQueries.time_column_hist(column=column.name, table=table.name))
                )

                column.stats.values.hist = hist
                column.stats.values.freq = freq
            elif column.data_type == DataType.STRING:
                column.stats.values = cast(StringStats, column.stats.values)

                len_hist, len_freq = results_to_hist_freq(
                    self.execute(StatQueries.str_len_column_hist(column=column.name, table=table.name))
                )

                column.stats.values.length.hist = len_hist
                column.stats.values.length.freq = len_freq

                word_hist, word_freq = results_to_hist_freq(
                    self.execute(StatQueries.str_word_len_column_hist(column=column.name, table=table.name))
                )

                column.stats.values.word.hist = word_hist
                column.stats.values.word.freq = word_freq
            else:
                raise NotImplementedError()


def model_from_create_commands(file) -> Database:
    with open(file) as f:
        text = ''.join(f.readlines())

    # sql to_relational_tree can read create tables ..
    # TODO:should really replace this with a simpler piece of code
    commands = sqlparse.parse(text)

    tables = []
    for command in commands:
        columns = []
        for token in command.tokens:
            if type(token) is sqlparse.sql.Identifier and type(token.parent) is sqlparse.sql.Statement:
                table_name = str(token)
            if type(token) is sqlparse.sql.Parenthesis and type(token.parent) is sqlparse.sql.Statement:
                column_defs = token.value[1:-1].replace('\n', ' ').strip().split(',')
                for column_def in column_defs:
                    column_name, column_type, *rest = column_def.strip().split(' ')
                    if column_type == 'character':
                        column_type += f' {rest[0]}'
                    columns.append(Column(column_name, to_known_data_type(column_type)))

        if command.value.replace('\n', '').strip().startswith('CREATE'):
            tables.append(Table(table_name, columns))

    return 9/Database(tables)
