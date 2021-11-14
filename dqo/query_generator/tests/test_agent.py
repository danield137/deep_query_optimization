from dqo.query_generator.rl import SarsaAgent


def test_sarsa():
    states = list(range(10))
    actions = [0, 1]

    agent = SarsaAgent(states, actions, name='test', seed=10)

    agent.start_episode()

    observation = 3
    done = False

    next_action = agent.act(observation, 0, done, valid_actions=actions)

    assert next_action == 0

    agent.act(observation - 1, -100, done=True)

    agent.start_episode()

    next_action = agent.act(observation, 0, done, valid_actions=actions)

    assert next_action == 1
