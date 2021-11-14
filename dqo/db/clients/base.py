import abc
from typing import Any, List, Optional, Tuple

from dqo.db.models import Database


class DatabaseClientException(Exception):
    pass


class QueryTimeoutException(DatabaseClientException):
    pass


class DatabaseClient(metaclass=abc.ABCMeta):
    _schema: Optional[Database]
    query_timeout_secs: int

    @abc.abstractmethod
    def humanize_target(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def execute(self, query: str, as_dict=True, collect=True) -> List[Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def time(self, query: str) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def analyze(self, query: str) -> Tuple[float, float, str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def model(self, use_cache=True) -> Database:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()
