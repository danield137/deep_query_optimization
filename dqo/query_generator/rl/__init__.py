from __future__ import annotations

import logging
import os
import time
from typing import List, cast, Set, Union

from dqo.db.clients import DatabaseClient
from .agent import SarsaAgent
from .envs import DatabaseEnvV1
from .partitioner import Log2Partitioner
from dqo import log_utils
from dqo.relational import Query

logger = logging.getLogger('query_generator.rl')
logger.setLevel(logging.INFO)


def rargmin(a: List[Union[int, float]]) -> int:
    mn = 1e10
    if not a:
        return -1
    mn_i = -1
    i = len(a) - 1
    while i >= 0:
        if a[i] < mn:
            mn = a[i]
            mn_i = i
        i -= 1

    return mn_i


# TODO: should inherit from QueryExecutor
class EpisodicQueryGen:
    """
    Using reinforcement learn, explore the query space,
    with the goal to advance in a uniform way, such that runtime
    distribution is uniform on the log2 plane.
    """
    mem: Set[Query]
    agents: List[SarsaAgent]
    partitions: List[int]
    counter: int = 0
    q: Query

    def __init__(self, db_client: DatabaseClient, ctx: dict = None):
        filename = os.path.join(
            'runtimes',
            f'{db_client.humanize_target()}_{str(int(time.time()))}.csv'
        )
        self._query_logger = log_utils.rotating_logger(filename=filename, ctx=ctx)
        self.db_env = DatabaseEnvV1(db_client, ctx)

        self.valid_actions = self.db_env.get_valid_actions()
        self.partitioner = Log2Partitioner(min_value=1, max_value=2 ** 8)
        self.states = list(range(self.partitioner.k))
        self.actions = list(range(len(self.db_env.get_action_space())))

        # we need k agents, for k targets
        self.agents = []

        for i in range(self.partitioner.k):
            a = SarsaAgent(self.states, self.actions, inital_actions=self.db_env.get_valid_actions(), name=str(i), ctx=ctx)
            state_file = f'agent_{a.name}.state'
            if os.path.exists(state_file):
                a.load(state_file)
            self.agents.append(a)

        self.partitions = [0] * self.partitioner.k

        logger.info(f'Initializing EpisodicQueryGen with {len(self.agents)} agents.')

    def record_query_execution(self, q: str, took: float):
        self._query_logger.debug(f'"{q}", {took}')
        self.partitions[self.partitioner.partition(took)] += 1

    def run_episode(self):
        # TODO: randomize from the minimal values

        target = cast(int, rargmin(self.partitions))
        logger.info(f'target: {target}')
        self.db_env.reset(target)

        _agent = self.agents[target]
        _agent.finish_episode()
        _agent.start_episode()

        done = False

        observation = -1
        reward = 0
        steps = 0
        total_reward = 0

        valid_actions = self.db_env.get_valid_actions()

        while not done:
            action = _agent.act(observation, reward, False, valid_actions)
            observation, reward, done, info = self.db_env.step(action, self.record_query_execution)
            steps += 1
            total_reward += reward
            valid_actions = info.get('next_actions')
            logger.debug(
                f'agent:{_agent.name} | '
                f'#{steps}) action: {action} ({self.db_env.get_action_desc(action)}), partition: {observation}, reward:{reward}, done:{done} '
                f'| total:{total_reward}, step:{steps}')

        # let the agent now we are done
        _agent.act(observation, reward, True)
        self.counter += steps
        _agent.finish_episode()

        logger.info(f'saving agent_{_agent.name}... (avg: {(total_reward / steps):.2f}, steps: {steps})')
        _agent.save(f'agent_{_agent.name}.state')
        return steps, total_reward

    def run(self, episodes=3):
        total_queries = 0
        total_reward = 0
        for i in range(episodes):
            logger.info(f'Episode {i}: Starting {self.partitions}')
            steps, accumulated_reward = self.run_episode()
            logger.info(f'Episode {i}: Done. took {steps} steps, with total reward {accumulated_reward} (avg: {(accumulated_reward / steps):.2f})')
            total_queries += steps
            total_reward += accumulated_reward
