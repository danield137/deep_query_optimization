import logging
import os
import time
from collections import defaultdict
from queue import Queue
from typing import Tuple, Callable, Dict, Optional

import numpy as np

from dqo import log_utils
from dqo.db.clients import DatabaseClient
from dqo.db.models import Database
from dqo.lab.query_executor import QueryExecutor
from dqo.query_generator import RandomQueryGen
from dqo.query_generator.query_builder import QueryBuilder
from dqo.query_generator.rl import rargmin
from dqo.query_generator.rl.partitioner import Partitioner, Log2Partitioner
from dqo.relational import Query, SQLParser

logger = logging.getLogger('query_generator.guided')
logger.setLevel(logging.INFO)

indexer = 0


class QueryGenError(Exception):
    pass


# todo: add cardinality considerations
class GuidedQueryGen(QueryExecutor):
    rqg: RandomQueryGen
    scheme: Database

    qb: QueryBuilder = None

    mem: Dict[Query, float] = defaultdict(float)
    cb: Callable[[Query, float], None]

    def __init__(
            self,
            db_client: DatabaseClient,
            target: Tuple[float, float],
            stop_early: bool = False,
            max_steps: int = 100,
            name: str = None,
            cb: Callable[[Query, float], None] = None,
            query_logger: logging.Logger = None,
            seed: int = None,
            extended: bool = False
    ):
        '''
        :param db_client:
        :param target: a tuple with (min, max) values to consider a "hit"
        :param stop_early: if true, returns done on first occurrence of a "hit"
        :param max_steps: limit number of steps
        :param cb: ball back to allow extra work on query, runtime tuple
        '''
        super().__init__(db_client, query_logger=query_logger, extended=extended)
        global indexer
        indexer += 1

        self.cb = cb
        self.stop_early = stop_early
        self.target = target
        self.steps = 0
        self.max_steps = max_steps if max_steps is not None else 100
        self.name = name or indexer
        self._rqg = None
        self._qb = None
        self._scheme = None
        self.seed = seed
        self.extended = extended

    @property
    def scheme(self) -> Database:
        if self._scheme is None:
            self._scheme = self.dbc.model()

        return self._scheme

    @property
    def qb(self) -> QueryBuilder:
        if self._qb is None:
            self._qb = QueryBuilder(self.scheme, seed=self.seed)
        return self._qb

    @property
    def rqg(self) -> RandomQueryGen:
        if self._rqg is None:
            self._rqg = RandomQueryGen(self.scheme)
        return self._rqg

    def run_query(self, analyze=True) -> Tuple[float, bool]:
        if self.qb.q not in self.mem:
            if analyze or self.extended:
                plan_time, exec_time, exec_plan = self.analyze(self.qb.q)
                took = plan_time + exec_time
            else:
                took = self.time(self.qb.q)

            query = self.qb.q.to_sql(pretty=False, alias=False)
            if self.cb and callable(self.cb):
                self.cb(self.qb.q, took)

            self.mem[self.qb.q] = took

        runtime = self.mem[self.qb.q]
        return runtime, self.hit(runtime)

    @property
    def current_sql(self) -> str:
        return self.qb.q.to_sql(pretty=False, alias=False)

    def randomize_initial(self):
        self.qb.q = self.rqg.randomize()
        self.qb.sync()

    def narrow(self):
        actions = []

        if self.qb.can_remove_projection():
            actions.append(self.qb.remove_projection)
        if self.qb.can_remove_relation():
            actions.append(self.qb.remove_relation)

        actions.append(self.qb.add_condition)

        action = np.random.choice(actions)
        action()

    def stay(self):
        actions = []

        if self.qb.can_remove_projection():
            self.qb.remove_projection()
        else:
            self.qb.add_projection()

    def broaden(self):
        actions = []

        if self.qb.can_add_projection():
            actions.append(self.qb.add_projection)
        if self.qb.can_add_relation():
            actions.append(self.qb.add_relation)
        if self.qb.can_remove_condition():
            actions.append(self.qb.remove_condition)
        if self.qb.can_replace_join():
            actions.append(self.qb.replace_join)

        if not actions:
            raise QueryGenError('no more options to broaden')
        action = np.random.choice(actions)
        action()

        # todo: add a stupid condition, like for a range [0,1], add > 0.1 and < 0.09,
        #  just to add another scan over the data
        #  generally this may not have much of an effect, but, for joins, it can wreck havoc

    def select_next_action(self, runtime):
        _min, _max = self.target
        if runtime > _max:
            return self.narrow
        elif runtime < _min:
            return self.broaden
        else:
            return self.stay

    def step(self, prev_runtime: float) -> Tuple[float, str, bool]:
        action = self.select_next_action(prev_runtime)
        action()

        runtime, hit = self.run_query()
        done = (self.stop_early and hit) or self.steps >= self.max_steps

        return runtime, action.__name__, done

    def hit(self, runtime: float):
        return self.target[0] <= runtime <= self.target[1]

    def guide(self):
        self.steps += 1
        runtime, done = self.run_query()
        while not done:
            prev_runtime = runtime
            runtime, action_took, done = self.step(runtime)
            logger.info(f'step: {self.steps - 1}, prev: {prev_runtime}, action: {action_took}, runtime: {runtime}')


class BalancedQueryGen:
    def __init__(
            self,
            db_client: DatabaseClient,
            partitioner: Partitioner = Log2Partitioner(min_value=1, max_value=2 ** 8),
            cb: Callable[[Query, float], None] = None,
            q_depth: int = 10,
            checkpoint: bool = False,
            patience: Optional[int] = 10,
            name_postfix: Optional[str] = None,
            extended=False
    ):
        self.partitioner = partitioner
        self.partitions = [0] * partitioner.k
        self.user_cb = cb
        self.checkpoint_path: str = f'{db_client.humanize_target()}.qcp'
        # use queue depth to counter postgres's caching, by checking last
        self.q: Queue[Tuple[GuidedQueryGen, float]] = Queue(q_depth)
        self.mem: Dict[Query, float] = defaultdict(float)
        self.patience = patience
        self.extended = extended

        def wrapped_cb(q: Query, runtime: float):
            if runtime > 0 and q not in self.mem:
                self.mem[q] = runtime
                partition = self.partitioner.partition(runtime)
                self.partitions[partition] += 1

            if cb is not None:
                cb(q, runtime)

        self.db_client = db_client
        self.wrapped_cb = wrapped_cb

        name_postfix = name_postfix or "_extended" if self.extended else ""
        filename = os.path.join(
            'runtimes',
            f'{db_client.humanize_target()}_{str(int(time.time()))}{name_postfix}.csv'
        )
        self.query_logger = log_utils.rotating_logger(filename=filename)
        self.checkpoint = checkpoint

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return

        with open(self.checkpoint_path) as cp:
            lines = cp.readlines()
            logger.info(f'loading {len(lines)} queries')
            if len(lines) > self.q.maxsize:
                self.q = Queue(maxsize=len(lines))

            for idx, line in enumerate(lines):
                try:
                    query = SQLParser.to_query(line)
                    # TODO: this is somewhat pointless, as we don't save the old distribution
                    min_partition = rargmin(self.partitions)
                    gqg = GuidedQueryGen(
                        db_client=self.db_client,
                        cb=self.wrapped_cb,
                        target=self.partitioner.bounds(min_partition),
                        # share logger
                        query_logger=self.query_logger,
                        extended=self.extended
                    )

                    gqg.qb.q = query
                    gqg.qb.sync()
                    self.q.put((gqg, 0))
                except:
                    pass

    def save_checkpoint(self):
        with open(self.checkpoint_path, 'w+') as cp:
            for qgq, _ in list(self.q.queue):
                cp.write(qgq.qb.q.to_sql(pretty=False, alias=False) + '\n')

    def generate(self, n: int = 100000):
        i = 0

        if self.checkpoint:
            self.load_checkpoint()

        tracking: Dict[int, Tuple[int, int]] = defaultdict(lambda x: (0, 0))

        logger.info('starting query generation loop')
        while len(self.mem.keys()) <= n:
            if self.checkpoint and i > 0 and i % self.q.maxsize == 0:
                self.save_checkpoint()

            if not self.q.full():
                min_partition = rargmin(self.partitions)

                self.q.put((
                    GuidedQueryGen(
                        db_client=self.db_client,
                        cb=self.wrapped_cb,
                        target=self.partitioner.bounds(min_partition),
                        # share logger
                        query_logger=self.query_logger
                    ), 0)
                )
                continue

            gqn, prev = self.q.get()
            i += 1

            done = True
            runtime = -1
            if prev == 0:
                if len(gqn.qb.q) == 0:
                    gqn.randomize_initial()
                try:
                    runtime, done = gqn.run_query()
                except Exception as e:
                    pass
            else:
                try:
                    runtime, _, done = gqn.step(prev)

                    if self.patience is not None:
                        prev_partition, seq_length = tracking[id(gqn)]
                        current_partition = self.partitioner.partition(runtime)

                        reset_count = current_partition != prev_partition
                        seq_length = 0 if reset_count else seq_length + 1

                        if seq_length > self.patience:
                            done = True
                            del tracking[id(gqn)]
                        else:
                            tracking[id(gqn)] = current_partition, seq_length
                except:
                    pass

            logger.info(f'[{gqn.name}:{gqn.steps}] - {runtime}')
            if not done:
                self.q.put((gqn, runtime))

            logger.info(f'BalancedQueryGen.generate | i:{i} - partitions: {self.partitions}')
