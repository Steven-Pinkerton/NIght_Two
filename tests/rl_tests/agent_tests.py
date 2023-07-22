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
        

if __name__ == '__main__':
    unittest.main()