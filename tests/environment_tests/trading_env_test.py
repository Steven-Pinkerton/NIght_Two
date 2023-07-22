import os
import unittest
import mock
import pandas as pd
import numpy as np
from collections import defaultdict
from pandas.testing import assert_frame_equal
from unittest.mock import patch
import pytest

from night_two.environment.trading_env import TradingEnvironment

class TestTradingEnvironment(unittest.TestCase):
    def setUp(self):
        # A small sample data frame for testing
        sample_data = pd.DataFrame({
            'Open': [1, 2, 4, 8],
            'High': [2, 3, 5, 10],
            'Low': [0.5, 1.5, 3.5, 7.5],
            'Close': [1.5, 2.5, 4.5, 9],
            'Volume': [100, 200, 300, 400]
        })

        sample_data.to_csv('sample_data.csv')

        # Create an instance of the trading environment
        self.trading_env = TradingEnvironment(initial_cash_balance=10000.0, 
                                            transaction_cost=0.01, 
                                            data_source='sample_data.csv')

        # Set an initial number of shares
        self.trading_env.num_shares = 300.00
        

        # Set initial portfolio value
        initial_stock_value = self.trading_env.num_shares * sample_data.loc[0, 'Close']
        self.trading_env.current_portfolio_value = self.trading_env.cash_balance + initial_stock_value
        self.trading_env.previous_portfolio_value = self.trading_env.current_portfolio_value
        
    def test_init(self):
        # Assert that the cash balance is initialized correctly
        self.assertEqual(self.trading_env.cash_balance, 10000.0)

        # Assert that the transaction cost is initialized correctly
        self.assertEqual(self.trading_env.transaction_cost, 0.01)

        # Assert that the data is loaded correctly
        self.assertEqual(len(self.trading_env.data), 4)

        # Assert that the action and observation spaces are defined
        self.assertIsNotNone(self.trading_env.action_space)
        self.assertIsNotNone(self.trading_env.observation_space)

    def test_initialize_attributes(self):
        # Assert that the indicators are initialized correctly
        self.assertIsNotNone(self.trading_env.INDICATORS)

        # Assert that the chosen_indicators dictionary is empty
        self.assertDictEqual(self.trading_env.chosen_indicators, {})

        # Assert that the params_values dictionary is initialized correctly
        self.assertDictEqual(self.trading_env.params_values, self.trading_env.INDICATORS)

        # Assert that the indicator_values dictionary is initialized correctly
        self.assertDictEqual(self.trading_env.indicator_values, 
                            {name: None for name in self.trading_env.INDICATORS.keys()})

        # Assert that the portfolio attributes are initialized correctly
        initial_stock_value = self.trading_env.num_shares * self.trading_env.data.loc[0, 'Close']
        expected_portfolio_value = self.trading_env.cash_balance + initial_stock_value

        self.assertEqual(self.trading_env.current_portfolio_value, expected_portfolio_value)
        self.assertEqual(self.trading_env.previous_portfolio_value, expected_portfolio_value)
        self.assertEqual(self.trading_env.historical_peaks, 0)
        self.assertEqual(self.trading_env.num_shares, 300.00)
        self.assertEqual(self.trading_env.total_trades, 0)

        # Assert that the return and value history lists are initialized as empty lists
        self.assertListEqual(self.trading_env.risk_adjusted_return_history, [])
        self.assertListEqual(self.trading_env.portfolio_value_history, [])

        # Assert that the state is initialized
        self.assertIsNotNone(self.trading_env.state)
           
    def test_define_observation_space(self):
        # Call the method with an example size
        obs_space = self.trading_env._define_observation_space(5)
        
        # Assert that the length of the observation space is correct
        self.assertEqual(len(obs_space), 5)

        # Assert that all elements in the observation space are None
        self.assertTrue(all(element is None for element in obs_space))

    def test_define_action_space(self):
        # Call the method
        action_space = self.trading_env._define_action_space()

        # Define the expected action space
        expected_action_space = ['buy', 'sell', 'hold', 'change_indicator_settings', 'select_indicators']

        # Assert that the action space is correct
        self.assertListEqual(action_space, expected_action_space)       
           
    def test_load_market_data(self):
        # Create a small dataframe for testing
        data = {'Close': [1, 2, 3, 4, 5], 'Open': [1.1, 2.1, 3.1, 4.1, 5.1]}
        df = pd.DataFrame(data)
        
        # Write this dataframe to a CSV file
        df.to_csv('test_data.csv', index=False)

        # Call the method with the test data
        loaded_data = self.trading_env.load_market_data('test_data.csv')
        
        # Assert that the loaded data is equal to the test data
        pd.testing.assert_frame_equal(loaded_data, df)
        
        # Cleanup: remove the test csv file
        os.remove('test_data.csv')       
           
    def test_reset(self):
        # Change some attributes from their initial state
        self.trading_env.current_step = 10
        self.trading_env.portfolio = 5000
        self.trading_env.cash_balance = 5000
        self.trading_env.buy_price = 100
        self.trading_env.sell_price = 200
        self.trading_env.winning_trades = 1
        self.trading_env.total_trades = 1

        # Call the reset method
        self.trading_env.reset()

        # Assert that all attributes have been reset to their initial state
        self.assertEqual(self.trading_env.current_step, 0)
        self.assertEqual(self.trading_env.portfolio, 0)
        self.assertEqual(self.trading_env.cash_balance, self.trading_env.initial_cash_balance)
        self.assertEqual(self.trading_env.buy_price, 0)
        self.assertEqual(self.trading_env.sell_price, 0)
        self.assertEqual(self.trading_env.winning_trades, 0)
        self.assertEqual(self.trading_env.total_trades, 0)       
           
    def test_step(self):
        # Define a small, known market data for testing
        data = {'Close': [1, 2, 3, 4, 5], 'Open': [1.1, 2.1, 3.1, 4.1, 5.1]}
        df = pd.DataFrame(data)

        # Initialize the trading environment with this known market data
        self.trading_env = TradingEnvironment(data_source=df)

        # Test the 'buy' action
        initial_state = self.trading_env.state.copy()
        action = {'type': 'buy', 'percentage': 100}
        new_state, reward, done = self.trading_env.step(action)
        self.assertFalse(np.array_equal(initial_state, new_state))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

        # Reset the environment
        self.trading_env.reset()

        # Test the 'sell' action
        initial_state = self.trading_env.state.copy()
        action = {'type': 'sell', 'percentage': 100}
        new_state, reward, done = self.trading_env.step(action)
        self.assertFalse(np.array_equal(initial_state, new_state))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

        # Reset the environment
        self.trading_env.reset()

        # Test the 'hold' action
        initial_state = self.trading_env.state.copy()
        action = {'type': 'hold'}
        new_state, reward, done = self.trading_env.step(action)
        self.assertFalse(np.array_equal(initial_state, new_state))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

        # Reset the environment
        self.trading_env.reset()

        # Test the 'change_indicator_settings' action
        initial_state = self.trading_env.state.copy()
        action = {'type': 'change_indicator_settings', 'settings': {'sma': {'period': 20}}}
        new_state, reward, done = self.trading_env.step(action)
        self.assertFalse(np.array_equal(initial_state, new_state))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

        # Reset the environment
        self.trading_env.reset()

        # Test the 'select_indicators' action
        initial_state = self.trading_env.state.copy()
        action = {'type': 'select_indicators', 'indicators': ['sma']}
        new_state, reward, done = self.trading_env.step(action)
        self.assertFalse(np.array_equal(initial_state, new_state))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)   
         
    def test_calculate_reward(self):
        # Define a small, known market data for testing
        data = {'Close': [1, 2, 3, 4, 5], 'Open': [1.1, 2.1, 3.1, 4.1, 5.1]}
        df = pd.DataFrame(data)

        # Initialize the trading environment with this known market data
        self.trading_env = TradingEnvironment(data_source=df)

        # Buy asset
        action = {'type': 'buy', 'percentage': 100}
        self.trading_env.step(action)

        # Compute reward after buying
        reward = self.trading_env.calculate_reward()
        self.assertIsInstance(reward, float)

        # Advance a few steps in the environment
        for _ in range(5):
            self.trading_env.step({'type': 'hold'})

        # Compute reward after holding
        reward_after_holding = self.trading_env.calculate_reward()
        self.assertIsInstance(reward_after_holding, float)

        # Check that reward does not change over time when holding
        self.assertEqual(reward, reward_after_holding)

        # Sell asset
        action = {'type': 'sell', 'percentage': 100}
        self.trading_env.step(action)

        # Compute reward after selling
        reward_after_selling = self.trading_env.calculate_reward()
        self.assertIsInstance(reward_after_selling, float)

        # Check that reward changes after selling
        self.assertNotEqual(reward_after_holding, reward_after_selling)
        
    def test_update_parameters(self):
        # Initialize the trading environment
        self.trading_env = TradingEnvironment()

        # Define the parameters for an indicator
        params = {
            "param1": range(1, 11),  # 1 to 10 inclusive
            "param2": range(5, 16),  # 5 to 15 inclusive
            "param3": 7  # not a range
        }

        # Call the update_parameters method for a mock indicator
        updated_params = self.trading_env.update_parameters("mock_indicator", params)

        # Check that the updated parameters are no longer ranges and are within the expected bounds
        self.assertIsInstance(updated_params["param1"], int)
        self.assertGreaterEqual(updated_params["param1"], 1)
        self.assertLessEqual(updated_params["param1"], 10)

        self.assertIsInstance(updated_params["param2"], int)
        self.assertGreaterEqual(updated_params["param2"], 5)
        self.assertLessEqual(updated_params["param2"], 15)

        # Check that parameters which were not ranges remain the same
        self.assertEqual(updated_params["param3"], 7)  
       
    def test_calculate_and_store_indicator_value(self):
        # Initialize the trading environment
        self.trading_env = TradingEnvironment()

        # Define a simple indicator function
        def mock_indicator(data, multiplier):
            return data * multiplier

        # Define the parameters for the mock indicator
        params = {"multiplier": 2}

        # Store some mock indicator data
        self.trading_env.indicator_data = 3

        # Call the calculate_and_store_indicator_value method with the mock indicator
        self.trading_env.calculate_and_store_indicator_value("mock_indicator", mock_indicator, params)

        # Check that the mock indicator value has been stored correctly
        self.assertEqual(self.trading_env.indicator_values["mock_indicator"], 6)

    def test_select_indicators(self):
        # Initialize the trading environment
        self.trading_env = TradingEnvironment()

        # Define a list of indicators to select
        indicators = ["indicator1", "indicator2"]

        # Add these indicators to the environment's INDICATORS dict with mock params
        self.trading_env.INDICATORS = {
            "indicator1": {"params": {"period": range(5, 10)}},
            "indicator2": {"params": {"period": range(10, 15)}}
        }

        # Call the select_indicators method with the chosen indicators
        self.trading_env.select_indicators(indicators)

        # Check that the chosen indicators and their parameters have been stored correctly
        self.assertIn("indicator1", self.trading_env.chosen_indicators)
        self.assertIn("indicator2", self.trading_env.chosen_indicators)

        # Check that an error is raised when an unknown indicator is chosen
        with self.assertRaises(ValueError):
            self.trading_env.select_indicators(["indicator3"])
  
    def test_buy_asset(self):
        # Initialize the trading environment
        self.trading_env = TradingEnvironment()

        # Set an initial cash balance
        self.trading_env.cash_balance = 1000

        # Mock the market state
        self.trading_env.market_state = {'Close': 50}

        # Call the buy_asset method
        self.trading_env.buy_asset(50)

        # Check that the cash balance is decreased correctly
        self.assertEqual(self.trading_env.cash_balance, 500)

        # Check that the correct number of shares were bought
        self.assertEqual(self.trading_env.num_shares, 10)

        # Check that the total number of trades was incremented
        self.assertEqual(self.trading_env.total_trades, 1)   
         
    def test_sell_asset(self):
        # Initialize the trading environment
        self.trading_env = TradingEnvironment()

        # Set an initial cash balance and number of shares
        self.trading_env.cash_balance = 1000
        self.trading_env.num_shares = 20
        self.trading_env.buy_price = 50

        # Mock the market state
        self.trading_env.market_state = {'Close': 60}

        # Call the sell_asset method
        self.trading_env.sell_asset(50)

        # Check that the cash balance is increased correctly
        self.assertEqual(self.trading_env.cash_balance, 1000 + 0.5*20*60)

        # Check that the correct number of shares were sold
        self.assertEqual(self.trading_env.num_shares, 10)

        # Check that the total number of trades was incremented
        self.assertEqual(self.trading_env.total_trades, 1)

        # Check that winning trades were counted correctly
        self.assertEqual(self.trading_env.winning_trades, 1)     

if __name__ == '__main__':
    unittest.main()