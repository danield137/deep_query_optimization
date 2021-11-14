from dqo.db.tests.datasets import employees_db_w_meta
from dqo.db.tests.fake_client import FakeClient
from ..rl.envs.db_env import DatabaseEnvV1

fake_db = FakeClient(
    db=employees_db_w_meta()
)


def test_smoke():
    env = DatabaseEnvV1(fake_db)

    actions = env.get_action_space()
    valid_actions = env.get_valid_actions()

    assert len(valid_actions) == 1
    for idx, action in enumerate(actions):
        assert env.get_action_desc(idx) == action.name

    next_state, reward, done, info = env.step(valid_actions[0])

    assert next_state == 0
    assert reward == -1.0
    assert done is False

    took, query, next_actions = info
    assert took > 0
    assert len(query) > len('SELECT') and type(query) is str
    assert next_actions == [0, 1, 2, 3, 5]
