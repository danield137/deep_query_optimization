import random
from unittest.mock import patch, mock_open

from dqo.db.tests.datasets import employees_db_w_meta
from dqo.db.tests.fake_client import FakeClient
from ..query_executor import FileQueryExecutor

fake_data = """
SELECT * FROM table
SELECT * FROM table2
SELECT * FROM table
SELECT * FROM table2
SELECT * FROM table3
"""

fake_path = "fake/path"
fake_existing_path = "fake/existing"
fake_logs = """"SELECT a,b,c FROM table", 0.23
"SELECT * FROM table", 0.1
"""


def test_happy_execute():
    expected = [1, 2, 3]
    fake_client = FakeClient(db=employees_db_w_meta(), execution_time=0.1, fake_result=expected)
    random.seed(1)
    q_exec = FileQueryExecutor(fake_client, fake_path, shuffle=True, log=False)
    with patch("os.listdir", return_value=['fake_file.csv']):
        with patch("builtins.open", mock_open(read_data=fake_data)):
            with patch.object(q_exec._query_logger, 'debug') as mocked_logger:
                result = q_exec.execute()

    assert len(mocked_logger.call_args_list) == 3
    assert len(q_exec.queued) == 0

    query_log = [c[0] for c in mocked_logger.call_args_list]
    expected = set([q for q in fake_data.split('\n') if q.strip()])
    for row in query_log:
        comma_index = row[0].rfind(',')
        q, timed = row[0][:comma_index], row[0][comma_index + 1:]
        assert q[1:-1] in expected
        assert float(timed) > 0


def test_remove_existing():
    expected = [1, 2, 3]
    fake_client = FakeClient(db=employees_db_w_meta(), execution_time=0.1, fake_result=expected)
    random.seed(1)
    q_exec = FileQueryExecutor(fake_client, fake_path, existing_path=fake_existing_path, shuffle=True, log=False)
    with patch("os.listdir", return_value=['fake_file.csv']):
        with patch("builtins.open", mock_open(read_data=fake_data)):
            with patch.object(q_exec._query_logger, 'debug') as mocked_logger:
                q_exec.load_queries()

    assert len(q_exec.queued) == 3

    with patch("os.listdir", return_value=['fake_file.csv']):
        with patch("builtins.open", mock_open(read_data=fake_logs)):
            with patch.object(q_exec._query_logger, 'debug') as mocked_logger:
                q_exec.remove_existing()

    assert len(q_exec.queued) == 2
