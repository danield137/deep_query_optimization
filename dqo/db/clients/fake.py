import time
from typing import List, Any, Callable

from dqo.db.clients.base import DatabaseClient
from dqo.db.models import Database


class FakeClient(DatabaseClient):
    db: Database
    execution_time: float
    query_timeout_secs: int

    def __init__(self, db: Database, execution_time: float = 0, fake_result=None, fake_exec: Callable[[str], Any] = None):
        self.db = db
        self.execution_time = execution_time
        self.fake_result = fake_result
        self.fake_exec = fake_exec
        self.query_timeout_secs = 0

    def execute(self, query, as_dict=True, collect=True) -> List[Any]:
        if self.fake_exec and callable(self.fake_exec):
            return self.fake_exec(query)
        if self.execution_time > 0:
            time.sleep(self.execution_time)

        return self.fake_result

    def humanize_target(self) -> str:
        return "fake"

    def get_columns_stats_query(self, database) -> str:
        return "None"

    def model(self, use_cache=True) -> Database:
        return self.db

    def reset(self):
        pass

    def time(self, query: str) -> float:
        return 0
