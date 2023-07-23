import copy
import numpy as np
import unittest
import torch
from night_two.Reinforcement_Learning.agent import DDPGAgent, OUNoise, ReplayBuffer
from night_two.environment.trading_env import TradingEnvironment

class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()  # You need to replace this with the actual instantiation of your environment

        # Calculate state_dim and action_dim from the environment
        num_indicators = len(cls.env.chosen_indicators)
        cls.state_dim = num_indicators * 2  # Considering both parameters and values for each indicator
        cls.action_dim = len(['Buy', 'Sell', 'Hold', 'change_indicator_settings', "select_indicators"])  # Assuming these are all possible actions

        # Other hyperparameters
        cls.max_action = 1.0  # Assuming the action space is normalized between -1.0 and 1.0
        cls.lstm_hidden_dim = 32
        cls.num_lstm_layers = 1
        cls.dropout_rate = 0.1
        cls.max_buffer_size = 10000

        # Initialize the DDPGAgent
        cls.agent = DDPGAgent(cls.state_dim, cls.action_dim, cls.max_action, cls.lstm_hidden_dim, cls.num_lstm_layers, cls.dropout_rate, cls.max_buffer_size)

        # Generate a sample state
        cls.state = np.random.random((1, cls.state_dim))

    def test_get_action(self):
        action = self.agent.get_action(self.state)
        self.assertIsInstance(action, np.ndarray), "The output of get_action must be a numpy array"
        self.assertEqual(action.shape, (self.agent.action_dim,)), "The shape of the action must match the action_dim"

class TestOUNoise(unittest.TestCase):
    @classmethod
    def setUp(self):
        state_dim = 4
        action_dim = 2
        max_action = 1.0
        lstm_hidden_dim = 64
        num_lstm_layers = 1
        dropout_rate = 0.1
        max_buffer_size = 10000
        self.action = np.array([0.5, 0.5])  # Define your action here
        self.state = np.ones(state_dim)  # Define your state here, using np.ones for simplicity

        self.agent = DDPGAgent(state_dim, action_dim, max_action, lstm_hidden_dim, num_lstm_layers, dropout_rate, max_buffer_size)
        self.noise = OUNoise(action_dim)  # replace with actual arguments
        
    def test_noise_addition(self):
        self.noise.reset()
        noise = self.noise.evolve_state()
        expected_noisy_action = self.action + noise
        actual_noisy_action = self.agent.select_action(self.state, noise=True)

        # Check if the noise is within the expected range
        lower_bound = self.action - 1.0  # Change these bounds as necessary
        upper_bound = self.action + 1.0

        # Reshape the upper and lower bounds to match the actual_noisy_action shape
        lower_bound = np.reshape(lower_bound, actual_noisy_action.shape)
        upper_bound = np.reshape(upper_bound, actual_noisy_action.shape)

        np.testing.assert_array_less(actual_noisy_action, upper_bound)
        np.testing.assert_array_less(lower_bound, actual_noisy_action)

class TestReplayBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state_dim = 4
        cls.buffer_size = 10
        cls.batch_size = 2

        # Initialize the ReplayBuffer
        cls.buffer = ReplayBuffer(cls.buffer_size)  # corrected this line

    def test_add_sample(self):
        state = np.ones(self.state_dim)
        action = np.array([0.5])
        reward = 1.0
        next_state = np.ones(self.state_dim) * 2
        done = False

        self.buffer.add(state, action, reward, next_state, done)

        # Check if the added sample is in the buffer
        self.assertEqual(len(self.buffer), 1, "The buffer length must increase by 1 after adding a sample")

        stored_state, stored_action, stored_reward, stored_next_state, stored_done = self.buffer.buffer[0]
        self.assertTrue(np.array_equal(stored_state, state), "The stored state must match the added state")
        self.assertTrue(np.array_equal(stored_action, action), "The stored action must match the added action")
        self.assertEqual(stored_reward, reward, "The stored reward must match the added reward")
        self.assertTrue(np.array_equal(stored_next_state, next_state), "The stored next state must match the added next state")
        self.assertEqual(stored_done, done, "The stored done flag must match the added done flag")

    def test_sample(self):
        # Fill the buffer
        for _ in range(self.buffer_size):
            state = np.random.rand(self.state_dim)
            action = np.random.rand(1)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_dim)
            done = bool(np.random.randint(0, 2))

            self.buffer.add(state, action, reward, next_state, done)

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.buffer.sample(self.batch_size)

        self.assertEqual(len(batch_states), self.batch_size, "Batch of states must have correct size")
        self.assertEqual(len(batch_actions), self.batch_size, "Batch of actions must have correct size")
        self.assertEqual(len(batch_rewards), self.batch_size, "Batch of rewards must have correct size")
        self.assertEqual(len(batch_next_states), self.batch_size, "Batch of next_states must have correct size")
        self.assertEqual(len(batch_dones), self.batch_size, "Batch of dones must have correct size")
        
    def test_buffer_size_limit(self):
        # Add more samples than the buffer size
        for _ in range(self.buffer_size + 5):
            state = np.random.rand(self.state_dim)
            action = np.random.rand(1)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_dim)
            done = bool(np.random.randint(0, 2))

            self.buffer.add(state, action, reward, next_state, done)

        # Check if buffer size does not exceed its limit
        self.assertEqual(len(self.buffer), self.buffer_size, "Buffer size must not exceed its limit")

class TestLearningProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state_dim = 4
        cls.action_dim = 2
        cls.max_action = 1.0
        cls.lstm_hidden_dim = 64
        cls.num_lstm_layers = 1
        cls.dropout_rate = 0.1
        cls.max_buffer_size = 10000
        cls.batch_size = 64

        # Initialize the DDPGAgent and ReplayBuffer
        cls.agent = DDPGAgent(cls.state_dim, cls.action_dim, cls.max_action, cls.lstm_hidden_dim, cls.num_lstm_layers, cls.dropout_rate, cls.max_buffer_size)
        cls.buffer = ReplayBuffer(cls.max_buffer_size)

    def test_single_learning_iteration(self):
        # Add enough samples to the replay buffer
        for _ in range(self.batch_size):
            state = np.random.rand(self.state_dim)
            action = np.random.rand(self.action_dim)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_dim)
            done = bool(np.random.randint(0, 2))

            self.buffer.add(state, action, reward, next_state, done)

        # Perform a learning iteration
        try:
            self.agent.learn(self.batch_size)
        except Exception as e:
            self.fail(f"Learning iteration raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()