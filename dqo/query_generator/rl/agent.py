import os
import time
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dqo import log_utils


class Agent(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def act(self, observation, reward, done):
        raise NotImplementedError()


class SarsaAgent(Agent):
    """
    Monte Carlo (alpha) Agent
    """
    q_table: np.array
    visits: np.array
    last_action: int
    # TODO: fix this, should not assume action type
    initial_actions: List[int]
    last_state: int
    target: int
    steps: int

    def __init__(
        self,
        states,
        actions,
        inital_actions=None,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed: int = None,
        name: str = None,
        log: bool = True,
        ctx: dict = None
    ):
        self.action_space = actions
        self.states = states

        self.initial_actions = inital_actions

        if seed is not None:
            np.random.seed(seed)

        self.alpha = alpha
        self.gamma = gamma
        # chance of exploration
        self.epsilon = epsilon

        self.name = name
        self.last_state = -1
        self.last_action = -1
        self.steps = 0
        self.visits = None
        self.q_table = None
        self.episode = 0

        self.q_table = np.zeros((len(self.states), len(self.action_space)), float)
        self.log = log
        if log:
            filename = os.path.join(
                'rl',
                f'{self.__class__.__name__}_{self.name}_{time.time()}.csv'
            )
            self.logger = log_utils.rotating_logger(filename=filename, ctx=ctx)

    def start_episode(self):
        self.visits = np.zeros((len(self.states), len(self.action_space)), int)
        self.steps = 0
        self.last_state = -1
        self.last_action = -1
        self.episode += 1

    def finish_episode(self):
        pass

    def learn(self, state, action, reward):
        if self.log and self.logger:
            self.logger.info(f'{self.episode},{self.steps},{self.last_state},{self.last_action},{reward}')

        predicted_state = self.q_table[self.last_state][self.last_action]
        target = reward + self.gamma * self.q_table[state][action]
        self.q_table[self.last_state][self.last_action] = self.q_table[self.last_state][self.last_action] + self.alpha * (target - predicted_state)
        self.visits[state][action] += 1

    def act(self, observation: int, reward: float, done: bool, valid_actions: List[int] = None) -> int:
        if self.steps == 0 and self.initial_actions is not None:
            valid_actions = self.initial_actions
        if observation < 0:
            next_action = np.random.choice(valid_actions) if valid_actions else np.random.randint(0, len(self.action_space))
        else:
            next_action = self.use_policy(observation, valid_actions)

        if self.last_state >= 0:
            self.learn(observation, next_action, reward)

        # invalid action
        self.last_state = observation
        self.last_action = next_action

        self.steps += 1

        return next_action

    def use_policy(self, observation: int, valid_actions: List[int] = None) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(valid_actions) if valid_actions else np.random.randint(0, len(self.action_space))
        elif valid_actions:
            max_actions = []
            max_action_score = -1e9
            for valid_action in valid_actions:
                if self.q_table[observation][valid_action] > max_action_score:
                    max_actions = [valid_action]
                    max_action_score = self.q_table[observation][valid_action]
                elif self.q_table[observation][valid_action] == max_action_score:
                    max_actions.append(valid_action)

            action = np.random.choice(max_actions)
        else:
            action = np.argmax(self.q_table[observation])

        return action

    def load(self, state_file: str):
        self.q_table = np.loadtxt(state_file)

    def save(self, state_file: str):
        np.savetxt(state_file, self.q_table)
