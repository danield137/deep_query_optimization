from enum import IntEnum
from typing import List, TypeVar

import numpy as np


class ChoicePreference(IntEnum):
    Uniform = 0
    Left = 1
    Right = 2


def randint_with_preference(a, b, preference: ChoicePreference) -> int:
    return choose_with_preference(list(range(a, b + 1)), preference)


T = TypeVar('T')


def choose_with_preference(
    population: List[T],
    preference: ChoicePreference
) -> T:
    n = len(population)

    probabilities = None

    if preference == ChoicePreference.Uniform:
        probabilities = [1 / n] * n
    else:
        values = np.array(list(range(1, n + 1)))
        norm = values / sum(values)
        if preference == ChoicePreference.Left:
            probabilities = norm[::-1]
        if preference == ChoicePreference.Right:
            probabilities = norm[:]

    return np.random.choice(population, p=probabilities)
