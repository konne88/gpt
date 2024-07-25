from dataclasses import dataclass
import math
from typing import Callable, Iterator, List, Tuple, TypeVar

Token = int


def softmax(list: List[float]) -> List[float]:
    return normalize([math.exp(element) for element in list])


def normalize(list: List[float]) -> List[float]:
    total = sum(list)
    return [element / total for element in list]


def weightedAverage(weights: List[float], values):
    return sum([weight * value for weight, value in zip(weights, values)])


A = TypeVar("A")
def prefixes(seq: List[A]) -> Iterator[List[A]]:
    for i in range(len(seq)):
        yield seq[:i + 1]

