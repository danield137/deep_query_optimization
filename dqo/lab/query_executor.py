import json
import logging.handlers
import os
import random
import numpy as np
from abc import ABC
from logging import Logger
from timeit import time
from typing import List, Tuple, Set, Union, Any

from tqdm import trange, tqdm

from dqo.datasets import QueriesDataset
from dqo import log_utils
from dqo.db.clients import DatabaseClient
from dqo.relational import Query

logger = logging.getLogger('lab.executor')
logger.setLevel(logging.DEBUG)


def clean(query: str) -> str:
    """ cleans strings from new lines and wrapping whitespaces"""
    return query.replace('"', '""').replace("\r\n", " ").replace("\n", " ").strip()


class QueryExecutor(ABC):
    _query_logger: Logger
    dbc: DatabaseClient
    limit: int

    def __init__(
            self,
            db_client: DatabaseClient,
            limit: int = -1,
            ctx: dict = None,
            query_logger: Logger = None,
            extended=False
    ):
        """
        :param db_client: any client of type @DatabaseClient
        """
        filename = os.path.join(
            'runtimes',
            f'{db_client.humanize_target()}_{str(int(time.time()))}{"_extended" if extended else ""}.csv'
        )

        self._query_logger = query_logger or log_utils.rotating_logger(filename=filename, ctx=ctx)
        self.dbc = db_client
        self.limit = limit

    def time(self, query: Union[Query, str]) -> float:
        self.dbc.execute("DEALLOCATE ALL", collect=False)
        self.dbc.execute("DISCARD PLANS", collect=False)
        self.dbc.execute("DISCARD TEMP", collect=False)

        query = query.to_sql(pretty=False, alias=False) if isinstance(query, Query) else query
        took = self.dbc.time(query) / 1000.0

        self._query_logger.debug(f'"{query}", {took}')

        return took

    def analyze(self, query: Union[Query, str]) -> Tuple[float, float, str]:
        self.dbc.execute("DEALLOCATE ALL", collect=False)
        self.dbc.execute("DISCARD PLANS", collect=False)
        self.dbc.execute("DISCARD TEMP", collect=False)

        query = query.to_sql(pretty=False, alias=False) if isinstance(query, Query) else query
        plan_time, exec_time, exec_plan = self.dbc.analyze(query)
        exec_time /= 1000.0
        plan_time /= 1000.0

        escaped_plan = json.dumps(exec_plan).replace('"', '""')
        self._query_logger.debug(f'"{query}",{exec_time + plan_time},"{escaped_plan}"')

        return plan_time, exec_time, exec_plan

    def execute(self, query: Union[Query, str], twice=True) -> Tuple[List[Tuple], float]:
        """
        Execute twice by default to canal out cache effects
        :param query:
        :param twice:
        :return:
        """
        self.dbc.execute("DEALLOCATE ALL", collect=False)
        self.dbc.execute("DISCARD PLANS", collect=False)
        self.dbc.execute("DISCARD TEMP", collect=False)

        query = query.to_sql(pretty=False, alias=False) if isinstance(query, Query) else query

        start = time.time()
        results = self.dbc.execute(query=query)
        took = time.time() - start

        exceeded_timeout = took < self.dbc.query_timeout_secs
        if twice and not exceeded_timeout:
            start = time.time()
            results = self.dbc.execute(query=query)
            took = time.time() - start

        self._query_logger.debug(f'"{query}", {took}')
        return results, took


class FileQueryExecutor(QueryExecutor):
    queries_path: str
    existing_path: str
    shuffle: bool

    rerun: bool
    queued: Set[str]

    def __init__(
            self,
            db_client: DatabaseClient,
            queries_path: str,
            existing_path: str = None,
            shuffle: bool = True,
            limit: int = -1,
            log=True
    ):
        """
        :param queries_path: path to a directory containing files with query-per-line
        :param existing_path: if provided, will remove the existing queries from the queries to run.
        :param shuffle: should shuffle the order of the queries prior to running (should ensure runs are fair)
        """

        super().__init__(db_client, limit)
        self.queries_path = queries_path
        self.existing_path = existing_path
        self.shuffle = shuffle

        self.queued = set()
        self.executed = set()

    def load_queries(self):
        files = os.listdir(self.queries_path)
        logger.info(f'found {len(files)} in "{self.queries_path}".')
        for file in files:
            file_path = os.path.join(self.queries_path, file)
            logger.info(f'reading queries from {file_path}')
            prev = len(self.queued)
            count = 0
            with open(file_path) as f:
                line = f.readline()
                while line:
                    clean_line = clean(line)
                    if clean_line:
                        count += 1
                        self.queued.add(clean_line)
                    line = f.readline()

            logger.info(f'found {count} (distinct: {len(self.queued) - prev}) in {file_path}')

    def remove_existing(self):
        if not self.existing_path:
            raise RuntimeError('not existing queries path given')

        files = os.listdir(self.existing_path)
        logger.info(f'found {len(files)} existing logs in "{self.queries_path}".')
        for file in files:
            logger.info(f'reading existing run from {file}".')
            file_path = os.path.join(self.existing_path, file)
            prev = len(self.queued)

            with open(file_path) as f:
                line = f.readline()
                while line:
                    # todo: better character escaping
                    comma_index = line.rfind(',')
                    query = line[:comma_index][1:-1]
                    clean_query = clean(query)
                    if clean_query in self.queued:
                        self.queued.remove(clean_query)
                    line = f.readline()

            logger.info(f'removed {prev - len(self.queued)}) queries, based on {file_path}')

        logger.info(f'total queries after removal: {len(self.queued)}')

    def execute(self, *args, **kwargs):
        if len(self.queued) == 0:
            self.load_queries()

        if self.existing_path:
            self.remove_existing()

        q = list(self.queued)

        if self.shuffle:
            random.shuffle(q)

        queries_count = len(q)
        logger.info(f'executing {queries_count} queries')

        for _ in trange(self.limit if self.limit > 0 else queries_count):
            query = q.pop()
            self.queued.remove(query)
            try:
                super().execute(query)
            except Exception as e:
                logger.warn(f"failed to run query\n {query}\n with error: {str(e)}")


class DatasetExecutor(QueryExecutor):
    queries_path: str
    existing_path: str
    shuffle: bool
    checkpoint_file: str = 'exec.cpf'

    rerun: bool

    def __init__(
            self,
            db_client: DatabaseClient,
            ds: QueriesDataset,
            shuffle: bool = False,
            limit: int = -1,
            log=True,
            extended=True,
            checkpoint=True
    ):
        """
        :param queries_path: path to a directory containing files with query-per-line
        :param existing_path: if provided, will remove the existing queries from the queries to run.
        :param shuffle: should shuffle the order of the queries prior to running (should ensure runs are fair)
        """

        super().__init__(db_client, limit, extended=extended)
        self.ds = ds
        self.shuffle = shuffle
        self.queries_df = None
        self.extended = extended
        self.checkpoint = checkpoint
        self.checkpoint_file = f'{db_client.humanize_target()}.cpf'

    def load_queries(self):
        self.queries_df = self.ds.load()

    def execute(self, *args, **kwargs):
        if self.queries_df is None or len(self.queries_df) == 0:
            self.load_queries()

        start_idx = kwargs.get('start_idx', 0)

        if self.checkpoint and os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as cpf:
                try:
                    start_idx = int(cpf.readline().strip())
                    print(f'resuming from {start_idx}')
                except:
                    raise ValueError(f'expected checkpoint file ({self.checkpoint_file}) to exist, and have a valid index.')

        queries = list(self.queries_df['query'])
        timings = list(self.queries_df['runtime'])
        end_idx = len(queries) + 1 if self.limit <= 0 else min(start_idx + self.limit, len(queries))

        queries = queries[start_idx:end_idx]
        timings = timings[start_idx:end_idx]
        # handle previous run
        logger.info(f'executing {len(queries)} queries {"with extended output" if self.extended else ""}')
        diff = []
        if self.checkpoint:
            with open(self.checkpoint_file, '+w') as cpf:
                for i, q in enumerate(tqdm(queries)):
                    try:
                        if self.extended:
                            plan_time, exec_time, exec_plan = super().analyze(q)
                            old_time = timings[i]
                            diff.append(np.abs(old_time - (exec_time + plan_time)))
                            if len(diff) == 100:
                                print(np.average(diff))
                                diff = []
                        else:
                            super().time(q)
                        cpf.write(str(start_idx + i))
                        cpf.seek(0)
                    except Exception as e:
                        logger.warning(f"failed to run query\n {q}\n with error: {str(e)}")
        else:
            for i, q in enumerate(tqdm(queries)):
                try:
                    if self.extended:
                        plan_time, exec_time, exec_plan = super().analyze(q)
                        old_time = timings[i]
                    else:
                        super().time(q)
                except Exception as e:
                    logger.warning(f"failed to run query\n {q}\n with error: {str(e)}")
