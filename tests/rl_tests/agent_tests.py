import copy

import numpy as np
from night_two.Reinforcement_Learning.agent import DDPGAgent, OUNoise, ReplayBuffer


def test_model_compilation():
    dummy_state_dim = 10
    dummy_action_dim = 2
    dummy_max_action = 1.0
    dummy_lstm_hidden_dim = 256
    dummy_num_lstm_layers = 2
    dummy_dropout_rate = 0.5
    dummy_max_buffer_size = 1000000

    agent = DDPGAgent(dummy_state_dim, dummy_action_dim, dummy_max_action, dummy_lstm_hidden_dim,
                      dummy_num_lstm_layers, dummy_dropout_rate, dummy_max_buffer_size)

    dummy_state = torch.randn(1, 1, dummy_state_dim)  # 1 batch, 1 sequence length
    assert agent.get_action(dummy_state) is not None

    dummy_action = torch.randn(1, 1, dummy_action_dim)  # 1 batch, 1 sequence length
    dummy_reward = torch.randn(1, 1)  # 1 batch, 1 sequence length
    dummy_next_state = torch.randn(1, 1, dummy_state_dim)  # 1 batch, 1 sequence length
    dummy_done = torch.tensor([[False]])  # 1 batch, 1 sequence length

    agent.store_transition(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
    agent.learn(1)
    
    
def test_replay_buffer():
    buffer_size = 10
    buffer = ReplayBuffer(buffer_size)

    state_dim = 4
    action_dim = 2
    state = np.ones(state_dim)
    action = np.ones(action_dim)
    reward = 1.0
    next_state = np.ones(state_dim)
    done = False

    # Add some experience to the buffer
    for _ in range(buffer_size):
        buffer.add(state, action, reward, next_state, done)

    # Sample some experience from the buffer
    batch_size = 5
    sampled_experience = buffer.sample(batch_size)

    # Check that the returned experience has the right shape
    assert len(sampled_experience) == 5
    assert sampled_experience[0].shape == (batch_size, state_dim)
    assert sampled_experience[1].shape == (batch_size, action_dim)
    assert len(sampled_experience[2]) == batch_size
    assert sampled_experience[3].shape == (batch_size, state_dim)
    assert len(sampled_experience[4]) == batch_size
    
    
def test_interaction_with_environment():
    env = YourEnvironment()
    agent = DDPGAgent(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.shape[0],
                      max_action=env.action_space.high[0],
                      lstm_hidden_dim=256,
                      num_lstm_layers=2,
                      dropout_rate=0.5,
                      max_buffer_size=1000000)
    state = env.reset()
    action = agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, done)    

def test_noise_process():
        action_dim = 2
        noise_process = OUNoise(action_dim)

        # Generate some noise
        initial_noise = noise_process.get_action(np.zeros(action_dim))

        # Decay the noise
        for _ in range(noise_process.decay_period):
            noise_process.get_action(np.zeros(action_dim), t=1)

        # Generate some more noise
        final_noise = noise_process.get_action(np.zeros(action_dim))

        # Check that the noise has decayed
        assert np.abs(final_noise).mean() < np.abs(initial_noise).mean()
        
def test_soft_update():
        state_dim = 10
        action_dim = 2
        max_action = 1.0
        lstm_hidden_dim = 256
        num_lstm_layers = 2
        dropout_rate = 0.5
        agent = DDPGAgent(state_dim, action_dim, max_action, lstm_hidden_dim, num_lstm_layers, dropout_rate, max_buffer_size=1000000)

        initial_weights = copy.deepcopy(agent.target_actor_model.state_dict())

        # Perform a soft update
        agent.soft_update(agent.actor_model, agent.target_actor_model, tau=0.5)

        final_weights = agent.target_actor_model.state_dict()

        # Check that the weights have been updated
        for key in initial_weights:
            assert not torch.equal(initial_weights[key], final_weights[key])
            
def test_action_range():
    env = YourEnvironment()
    agent = DDPGAgent(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.shape[0],
                      max_action=env.action_space.high[0],
                      lstm_hidden_dim=256,
                      num_lstm_layers=2,
                      dropout_rate=0.5,
                      max_buffer_size=1000000)

    # Run the agent for a few steps
    state = env.reset()
    for _ in range(100):
        action = agent.get_action(state)
        assert env.action_space.low[0] <= action <= env.action_space.high[0]
        state, _, _, _ = env.step(action)
        
def test_memory_buffer():
    buffer_size = 10
    buffer = ReplayBuffer(buffer_size)

    state_dim = 4
    action_dim = 2
    state = np.ones(state_dim)
    action = np.ones(action_dim)
    reward = 1.0
    next_state = np.ones(state_dim)
    done = False

    # Add some experiences to the buffer
    for i in range(buffer_size * 2):
        buffer.add(state * i, action, reward, next_state * i, done)

    # Check that the buffer size is correct
    assert len(buffer) == buffer_size

    # Check that the oldest experiences have been removed
    states, _, _, _, _ = buffer.sample(buffer_size)
    assert np.all([state[0] != 1 for state in states])