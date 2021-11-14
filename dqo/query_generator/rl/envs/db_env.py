import logging
import os
import time
from typing import Dict, Callable, NamedTuple, List

import numpy as np
from gym import Env
from gym.spaces import Discrete
from termcolor import colored

from dqo import log_utils
from dqo.db.clients import DatabaseClient
from dqo.query_generator.query_builder import QueryBuilder
from dqo.query_generator.rl.partitioner import Log2Partitioner
from dqo.relational import Query

logger = logging.getLogger('query_generator.rl')

StateType = int
ActionType = int
Info = NamedTuple('Info', [('took', float), ('query', str), ('next_actions', List[ActionType])])
StepResult = NamedTuple('StepResult', [('next_state', StateType), ('reward', float), ('done', bool), ('info', Info)])


class DatabaseEnvV1(Env):
    """
    DatabaseEnvV1 represents database.
    Agent's task is, given a target response time, to achieve that response time with a minimal amount of actions.

    ActionTypes:
        1. adding a relation
        2. adding a projection
        3. adding a condition
       // (optional) 4-6. remove corresponding action

    Reward:
        reward is given based on logarithmic distance from desired response time.

    Observation space:
        (state space) is defined as the possible logarithmic values.

    """
    BAD_STATE = -1

    cumulative_reward: float
    memory: Dict[Query, float]
    qb: QueryBuilder
    steps: int

    episode: int
    state: int

    partitioner = Log2Partitioner()
    observation_space = Discrete(Log2Partitioner().k)
    action_space = Discrete(6)

    def __init__(self, db_client: DatabaseClient, ctx: dict = None):
        self.dbc = db_client
        self.memory = {}
        self.cumulative_reward = 0
        self.steps = 0
        self.episode = 0
        self.target = DatabaseEnvV1.BAD_STATE
        self.state = DatabaseEnvV1.BAD_STATE
        self.qb = None

        filename = os.path.join(
            'rl',
            f'{self.__class__.__name__}.csv'
        )
        self.logger = log_utils.rotating_logger(name='db_env', filename=filename, ctx=ctx)

    def render(self, mode='human'):
        pass

    def get_action_space(self):
        if self.qb is None:
            schema = self.dbc.model()
            self.qb = QueryBuilder(schema)

        return self.qb.mutations[:]  # [m for m in self.qb.mutations if m.type == MutationType.Add]

    def get_valid_actions(self):
        if self.qb is None:
            schema = self.dbc.model()
            self.qb = QueryBuilder(schema)

        return self.qb.get_available_mutations()  # MutationType.Add)

    def execute(self, q: Query, twice=True):
        """
        Execute twice by default to canal out cache effects
        :param q:
        :param twice:
        :return:
        """
        logger.debug(colored(f'START executing query: {time.time()}', 'green'))

        start = time.time()
        self.dbc.execute(query=q.to_sql(alias=False))
        took = time.time() - start

        if twice:
            start = time.time()
            self.dbc.execute(query=q.to_sql(alias=False))
            took = time.time() - start

        logger.debug(colored(f'END executing query: {time.time()}', 'green'))
        return took

    def get_reward(self, took: float) -> float:
        partition = self.partitioner.partition(took)
        if partition == self.target:
            return 1000

        return -1 * (np.abs(partition - self.target) + (self.steps * 0.1))

    def get_observation(self, took: float) -> int:
        return self.partitioner.partition(took)

    def is_done(self, took: float):
        partition = self.partitioner.partition(took)
        return partition == self.target

    def step(self, action: ActionType, cb: Callable[[str, float], None] = None) -> StepResult:
        """
        returns new state as (next_state, reward, done, info_dict)
        """
        q = self.qb.mutate(action)
        logger.debug(f'\n{self.get_action_desc(action)} \n{"-" * 10}\n{q.to_sql(alias=False)}')

        observation = DatabaseEnvV1.BAD_STATE
        took = -1
        reward = -10
        done = False

        if not q.valid():
            self.qb.undo()
        else:
            try:
                if q not in self.memory:
                    took = self.execute(q)
                    self.memory[q] = took
                    if cb and callable(cb):
                        cb(q.to_sql(pretty=False, alias=False), took)
                else:
                    # no need to run again, we know the time
                    took = self.memory[q]

                observation = self.get_observation(took)
                reward = self.get_reward(took)
                done = self.is_done(took)

                self.cumulative_reward += reward
            except Exception as e:
                # took a bad turn
                print(str(e))

        if self.steps > 200:
            reward = -10000
            done = True

        self.steps += 1
        self.logger.info(f'{self.state},{self.get_action_desc(action)},{observation},{reward},{took},{q.to_sql(alias=False, pretty=False)}')

        self.state = observation
        return StepResult(
            observation, reward, done,
            Info(
                took=took,
                query=q.to_sql(alias=False, pretty=True),
                next_actions=self.get_valid_actions()
            )
        )

    def reset(self, target: StateType) -> StateType:
        self.qb.reset()
        self.cumulative_reward = 0
        self.target = target
        self.steps = 0
        self.episode += 1
        self.state = DatabaseEnvV1.BAD_STATE

        return self.state

    def get_action_desc(self, action: int) -> str:
        return self.qb.mutations[action].name