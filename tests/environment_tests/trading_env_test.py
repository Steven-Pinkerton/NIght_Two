import unittest
import pandas as pd
import numpy as np
from collections import defaultdict
from unittest.mock import patch
from unittest.mock import MagicMock
import pytest

from night_two.environment.trading_env import TradingEnvironment

class TestTradingEnvironment(unittest.TestCase):
    def setUp(self):
        # We'll create a default instance of the trading environment for each test
        self.env = TradingEnvironment()

    def test_init(self):
        # Test that the environment is initialized correctly
        self.assertEqual(self.env.cash_balance, 10000.0)
        self.assertEqual(self.env.transaction_cost, 0.01)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.shares, [])
        self.assertEqual(self.env.initial_cash_balance, 10000.0)

        # Check that all indicators have been initialized
        self.assertEqual(set(self.env.indicator_values.keys()), set(self.env.all_indicators.keys()))
        # All indicator values should be None at this point
        self.assertTrue(all(value is None for value in self.env.indicator_values.values()))

        # Check that chosen_indicators is an empty dictionary
        self.assertEqual(self.env.chosen_indicators, {})

        # Check that risk_adjusted_return_history and portfolio_value_history are empty lists
        self.assertEqual(self.env.risk_adjusted_return_history, [])
        self.assertEqual(self.env.portfolio_value_history, [])

        # Check that max_action has been calculated and is not None
        self.assertIsNotNone(self.env.max_action)

    @patch('pandas.DataFrame', autospec=True)
    def test_calculate_max_action(self, mock_df):
        # Mock the market_data attribute
        self.env.data = mock_df
        # Note: we should set the return_value of mock_df.columns to return a list
        mock_df.columns.__iter__.return_value = ["AAPL", "GOOG", "TSLA"]  # Let's assume we have 3 stocks for simplicity

        # Here, we can create some test scenarios to validate the max_action calculation
        expected_max_action = 3 * 4 * 10  # 3 stocks * 4 actions * 10 percentages
        actual_max_action = self.env.calculate_max_action()

        self.assertEqual(actual_max_action, expected_max_action)
        
    def test_initialize_state(self):
        # Mock market data
        self.env.data = pd.DataFrame({
            'AAPL': [150, 151, 152],
            'GOOG': [2700, 2701, 2702],
            'TSLA': [600, 601, 602]
        })
        self.env.current_step = 0
        self.env.initial_cash_balance = 10000.0

        # Mock all indicators
        self.env.all_indicators = {'sma': {'func': calculate_sma, 'params': {'period': range(10, 51)}}, 'rsi': {'func': calculate_rsi, 'params': {'period': range(5, 31)}}}

        # Call initialize_state
        returned_state = self.env.initialize_state()

        # Check the initial state of the environment
        self.assertEqual(self.env.num_shares, 0)
        self.assertEqual(self.env.cash_balance, 10000.0)
        self.assertEqual(self.env.market_state.tolist(), [150, 2700, 600])
        self.assertIsNot(self.env.chosen_indicators, self.env.all_indicators)
        self.assertEqual(self.env.previous_action, None)
        self.assertEqual(self.env.buy_price, 0)
        self.assertEqual(self.env.sell_price, 0)
        self.assertEqual(self.env.winning_trades, 0)
        self.assertEqual(self.env.total_trades, 0)

        # Check initial performance metrics
        # Assuming calculate_initial_metrics() returns {'portfolio_value': 10000.0, 'risk_adjusted_return': 0, 'winning_rate': 0, 'average_win': 0, 'average_loss': 0}
        self.assertEqual(self.env.performance_metrics, {'portfolio_value': 10000.0, 'risk_adjusted_return': 0, 'winning_rate': 0, 'average_win': 0, 'average_loss': 0})

        # Check the state vector
        # The state vector should be a concatenation of the portfolio vector, cash balance, market state, 
        # performance metrics, chosen indicators, and previous action. Each of these elements should be converted 
        # to the correct format before concatenation.

        expected_state = np.concatenate([
            np.zeros(3),  # portfolio vector (empty portfolio)
            np.array([10000.0]),  # cash balance
            np.array([150, 2700, 600]),  # market state
            np.array([10000.0, 0, 0, 0, 0]),  # performance metrics
            np.zeros_like(self.env.all_indicators),  # chosen indicators
            np.array([0])  # previous action
        ])

        np.testing.assert_array_equal(self.env.state, expected_state)
        np.testing.assert_array_equal(returned_state, expected_state)

    def test_concatenate_state(self):
        # Mock num_shares
        self.env.num_shares = 15

        # Mock cash balance
        self.env.cash_balance = 5000.0

        # Mock market state
        self.env.market_state = pd.Series([150, 2700, 600], index=['AAPL', 'GOOG', 'TSLA'])

        # Mock performance metrics
        self.env.performance_metrics = {
            'portfolio_value': 10000.0,
            'risk_adjusted_return': 0,
            'winning_rate': 0.6667,
            'average_win': 100,
            'average_loss': 50
        }

        # Mock chosen indicators
        self.env.chosen_indicators = {'sma': {'period': 30}, 'rsi': {'period': 14}}

        # Mock previous action
        self.env.previous_action = 1

        # Call concatenate_state
        state = self.env.concatenate_state()

        # The state vector should be a concatenation of the num_shares vector, cash balance, market state, 
        # performance metrics, chosen indicators, and previous action. Each of these elements should be converted 
        # to the correct format before concatenation.

        expected_state = np.concatenate([
            np.array([15]),  # num_shares vector
            np.array([5000.0]),  # cash balance
            np.array([150, 2700, 600]),  # market state
            np.array([10000.0, 0, 0.6667, 100, 50]),  # performance metrics
            self.indicator_settings_to_vector(self.env.chosen_indicators),  # chosen indicators
            np.array([1])  # previous action
        ])

        # Check the output state
        np.testing.assert_array_equal(state, expected_state)

    def test_metrics_to_vector(self):
        # Mock metrics
        metrics = {
            'portfolio_value': 10000.0,
            'risk_adjusted_return': 0,
            'winning_rate': 0.6667,
            'average_win': 100,
            'average_loss': 50
        }

        # Call metrics_to_vector
        vector = self.env.metrics_to_vector(metrics)

        # The output should be an array with the metric values, in the order they appear in the metrics dictionary
        expected_vector = np.array([10000.0, 0, 0.6667, 100, 50])

        # Check the output vector
        np.testing.assert_array_equal(vector, expected_vector)

    def test_action_to_vector(self):
        # Define action space
        self.env.action_space = ['Buy', 'Sell', 'Hold', 'Change Settings']

        # Define test scenarios
        test_scenarios = [
            ('Buy', np.array([0])),  # Action 'Buy' with index 0
            ('Sell', np.array([1])),  # Action 'Sell' with index 1
            (None, np.array([0])),  # None action, returns np.array([0]) by convention
            ('Hold', np.array([2])),  # Action 'Hold' with index 2
        ]

        # Run each test scenario
        for action, expected_vector in test_scenarios:
            # Call action_to_vector
            vector = self.env.action_to_vector(action)

            # Check the output vector
            np.testing.assert_array_equal(vector, expected_vector)
            
    def test_define_action_space(self):
        # Here, we can verify that the action space is correctly defined
        action_space = self.env.define_action_space()
        self.assertEqual(action_space, ['Buy', 'Sell', 'Hold', 'Change Settings'])
        self.assertEqual(self.env.action_space, action_space)
    
    def test_step_buy_action(self):
        # Assuming the action is represented as a string
        action = 'Buy'

        # Mock market state: AAPL price is $150
        self.env.market_state = pd.Series({'AAPL': 150.0})

        # Mock balance: $5000
        self.env.cash_balance = 5000.0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the balance was updated
        # This will depend on the logic inside your `update_portfolio_and_balance` function
        self.assertEqual(self.env.cash_balance, 5000 - 150)  # Assuming buying 1 share

        # Verify that the reward was calculated
        # This check will depend on the specifics of the reward calculation function
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)
    
    def test_step_sell_action(self):
        # Assuming the action is represented as a string
        action = 'Sell'

        # Mock portfolio: 10 shares of AAPL
        self.env.portfolio = {'AAPL': 10}

        # Mock market state: AAPL price is $200
        self.env.market_state = pd.Series({'AAPL': 200.0})

        # Mock balance: $5000
        self.env.cash_balance = 5000.0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the balance was updated
        self.assertEqual(self.env.cash_balance, 5000 + 200)  # Assuming selling 1 share

        # Verify that the reward was calculated 
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)

    def test_step_done(self):
        # Mock current step to be the last step
        self.env.current_step = len(self.env.market_data) - 1

        # Call the step function
        new_state, reward, done = self.env.step('Hold')

        # Verify that the episode has ended
        self.assertTrue(done)

        # Mock current cash balance to be 0
        self.env.cash_balance = 0.0

        # Call the step function
        new_state, reward, done = self.env.step('Hold')

        # Verify that the episode has ended
        self.assertTrue(done)
        
    def test_update_metrics_in_step_function(self):
        # Mock initial conditions
        self.env.portfolio = {'AAPL': 0}
        self.env.market_state = pd.Series([150.0], index=['AAPL'])
        self.env.cash_balance = 1000.0
        self.env.performance_metrics = {
            'Portfolio Value': 1000.0,
            'Running Average Value': 1000.0,
            'Drawdown': 0.0,
            'Winning Trades': 0,
            'Total Trades': 0
        }
        self.env.current_step = 0

        # Mock action: 'Buy'
        action = 'Buy'

        # Step the environment
        new_state, reward, done = self.env.step(action)

        # Verify that the metrics were updated correctly
        self.assertEqual(self.env.performance_metrics['Portfolio Value'], self.env.current_portfolio_value)
        # Assuming that 'Running Average Value' is updated in the update_metrics function
        self.assertEqual(self.env.performance_metrics['Running Average Value'], self.env.current_portfolio_value)
        # Assuming that 'Drawdown', 'Winning Trades', and 'Total Trades' are updated in the update_metrics function
        self.assertEqual(self.env.performance_metrics['Drawdown'], 0.0)
        self.assertEqual(self.env.performance_metrics['Winning Trades'], 0)
        self.assertEqual(self.env.performance_metrics['Total Trades'], 1)

    def test_episode_end_with_all_steps_taken(self):
        # Mock action: 'Buy'
        action = 'Buy'

        # Mock initial conditions
        self.env.cash_balance = 5000.0
        self.env.current_portfolio_value = 0
        self.env.market_data = pd.DataFrame({'AAPL': [200.0]})
        self.env.current_step = 0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the episode has ended
        self.assertTrue(done)

    def test_episode_end_with_no_balance(self):
        # Mock action: 'Buy'
        action = 'Buy'

        # Mock initial conditions
        self.env.cash_balance = 100.0  # not enough to buy 1 share of AAPL
        self.env.current_portfolio_value = 0
        self.env.market_data = pd.DataFrame({'AAPL': [200.0]})
        self.env.current_step = 0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the episode has ended
        self.assertTrue(done)
        
    def test_step_change_indicator_settings(self):
        # Mock action: 'Change Settings'
        action = {
            'type': 'change_indicator_settings',
            'indicator_name': 'sma', 
            'settings': {'parameter1': 10, 'parameter2': 5}
        }

        # Mock initial conditions
        self.env.cash_balance = 5000.0
        self.env.current_portfolio_value = 0
        self.env.market_data = pd.DataFrame({'AAPL': [200.0]})
        self.env.current_step = 0

        # Call the step function with the mock action
        new_state, reward, done = self.env.step(action)

        # Verify that the indicator settings were updated
        expected_settings = {'sma': {'parameter1': 10, 'parameter2': 5}}
        self.assertEqual(self.env.chosen_indicators, expected_settings)
  
    def test_step_previous_action(self):
        # Mock action: Buy
        action = {'type': 'buy', 'amount': 5}

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the previous action was stored correctly
        self.assertEqual(self.env.previous_action, action)
        
    def test_step_state_updates_portfolio(self):
        # Mock action: Buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Record the current portfolio value and cash balance before the action
        previous_portfolio_value = self.env.current_portfolio_value
        previous_cash_balance = self.env.cash_balance

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the portfolio value and cash balance were updated correctly
        self.assertNotEqual(self.env.current_portfolio_value, previous_portfolio_value)
        self.assertNotEqual(self.env.cash_balance, previous_cash_balance)

        # Verify that the update in portfolio value and cash balance is consistent with the action
        # Assuming 'close' is the closing price of the asset
        action_cost = action['amount'] * self.env.market_state['close']
        self.assertEqual(self.env.current_portfolio_value - previous_portfolio_value, action_cost)
        self.assertEqual(previous_cash_balance - self.env.cash_balance, action_cost)

    def test_step_updates_cash_balance_vector(self):
        # Mock action: Buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Get the cash_balance before the action
        previous_cash_balance = self.env.cash_balance

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the cash_balance was updated (the cash balance should be less now)
        self.assertTrue(self.env.cash_balance < previous_cash_balance)

    def test_step_updates_performance_vector(self):
        # Mock action: Buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Get the performance_metrics before the action
        previous_performance_metrics = self.env.performance_metrics.copy()

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the performance_metrics were updated
        self.assertNotEqual(self.env.performance_metrics, previous_performance_metrics)

    def test_step_updates_indicator_vector(self):
        # Mock action: change indicator settings
        action = {
            'type': 'change_indicator_settings',
            'settings': {'sma': {'period': 40}}
        }

        # Get the indicator_vector before the action
        previous_indicator_vector = self.env.indicator_settings_to_vector(self.env.chosen_indicators)

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the indicator_vector was updated
        self.assertFalse((self.env.indicator_settings_to_vector(self.env.chosen_indicators) == previous_indicator_vector).all())

    def test_step_updates_action_vector(self):
        # Mock action: buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Get the action_vector before the action
        previous_action_vector = self.env.action_to_vector(self.env.previous_action)

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the action_vector was updated
        self.assertNotEqual(self.env.action_to_vector(self.env.previous_action), previous_action_vector)

    def test_step_state_updates_cash_balance(self):
        # Mock action: buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Copy the current cash balance before the action
        previous_cash_balance = self.env.cash_balance

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the cash balance was updated (should decrease after a 'buy' action)
        self.assertNotEqual(self.env.cash_balance, previous_cash_balance)

    def test_step_state_updates_performance_metrics(self):
        # Mock action: buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Copy the current performance metrics before the action
        previous_performance_metrics = self.env.metrics_to_vector(self.env.performance_metrics).copy()

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the performance metrics were updated
        self.assertNotEqual(self.env.metrics_to_vector(self.env.performance_metrics), previous_performance_metrics)
            
    def test_step_state_updates_indicator_vector(self):
        # Mock action: change indicator settings
        action = {'type': 'change_indicator_settings', 'settings': {'sma': {'period': 40}}}

        # Copy the current indicator settings before the action
        previous_indicator_vector = self.env.indicator_settings_to_vector(self.env.chosen_indicators).copy()

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the indicator settings were updated
        self.assertNotEqual(self.env.indicator_settings_to_vector(self.env.chosen_indicators), previous_indicator_vector)

    def test_step_state_updates_action_vector(self):
        # Mock action: buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Copy the current action vector before the action
        previous_action_vector = self.env.action_to_vector(self.env.previous_action).copy()

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the action vector was updated
        self.assertNotEqual(self.env.action_to_vector(self.env.previous_action), previous_action_vector)

    def test_step_state_concatenation(self):
        # Mock action: buy 5 shares
        action = {'type': 'buy', 'amount': 5}

        # Copy the current state before the action
        previous_state = self.env.state.copy()

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the state was updated
        self.assertNotEqual(self.env.state, previous_state)

        # Verify that the state includes all components
        self.assertEqual(len(self.env.state), 
                        len(self.env.portfolio_to_vector(self.env.portfolio)) +
                        1 + len(self.env.market_state.values) +
                        len(self.env.metrics_to_vector(self.env.performance_metrics)) +
                        len(self.env.indicator_settings_to_vector(self.env.chosen_indicators)) +
                        len(self.env.action_to_vector(self.env.previous_action)))

    def test_valid_buy_action_updates_portfolio(self):
        self.env.cash_balance = 2000.0
        self.env.market_state = pd.Series({'symbol': 150.0})  # assuming 1 share costs $150
        action = {"type": "buy", "amount": 10}

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.num_shares, 10)
        self.assertEqual(self.env.cash_balance, 500.0)  # remaining cash balance after buying 10 shares

    def test_valid_sell_action_updates_portfolio(self):
        self.env.cash_balance = 2000.0
        self.env.num_shares = 10
        self.env.market_state = pd.Series({'symbol': 150.0})  # assuming 1 share costs $150
        self.env.buy_price = 100.0  # Mock buy price
        action = {"type": "sell", "amount": 5}

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.num_shares, 5)
        self.assertEqual(self.env.cash_balance, 2750.0)  # cash balance after selling 5 shares

    def test_insufficient_cash_prevents_buy(self):
        self.env.cash_balance = 1000.0
        self.env.market_state = pd.Series({'symbol': 150.0})  # assuming 1 share costs $150
        action = {"type": "buy", "amount": 10}

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.num_shares, 0)  # No shares should have been bought
        self.assertEqual(self.env.cash_balance, 1000.0)  # Cash balance should not have changed

    def test_insufficient_shares_prevents_sell(self):
        self.env.cash_balance = 2000.0
        self.env.num_shares = 3
        self.env.market_state = pd.Series({'symbol': 150.0})  # assuming 1 share costs $150
        action = {"type": "sell", "amount": 5}

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.num_shares, 3)  # No shares should have been sold
        self.assertEqual(self.env.cash_balance, 2000.0)  # Cash balance should not have changed

    def test_invalid_action_does_nothing(self):
        self.env.cash_balance = 2000.0
        self.env.num_shares = 0
        action = {"type": "fly", "amount": 10}  # Invalid action type

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.num_shares, 0)  # No shares should have been bought/sold
        self.assertEqual(self.env.cash_balance, 2000.0)  # Cash balance should not have changed
      
    def test_calculate_initial_metrics(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnvironment()
        
        # Manually set up the num_shares and market_state
        trading_env.num_shares = 10
        trading_env.market_state = pd.Series({'close': 100.0})

        # Call the calculate_initial_metrics function
        trading_env.calculate_initial_metrics()

        # Verify the calculated initial metrics
        self.assertEqual(trading_env.performance_metrics['Portfolio Value'], 1000.0)  # num_shares * market_state['close']
        self.assertEqual(trading_env.performance_metrics['Running Average Value'], 1000.0)  # num_shares * market_state['close']
        self.assertEqual(trading_env.performance_metrics['Drawdown'], 0)
        self.assertEqual(trading_env.performance_metrics['Winning Trades'], 0)
        self.assertEqual(trading_env.performance_metrics['Total Trades'], 0)
      
    def test_update_market_state(self):
        # Assuming self.env is an instance of your trading environment,
        # and it has been initialized with some mock market data.

        # Save the current step before calling update_market_state
        previous_step = self.env.current_step

        # Call update_market_state
        self.env.update_market_state()
        
        # If the previous step was less than the length of the market data minus one,
        # the current step should now be previous_step + 1
        if previous_step < len(self.env.market_data) - 1:
            self.assertEqual(self.env.current_step, previous_step + 1)
            self.assertEqual(self.env.market_state.all(), self.env.market_data.iloc[previous_step + 1].all())
        
        else:
            # If the previous step was not less than the length of the market data minus one,
            # the current step should still be the same
            self.assertEqual(self.env.current_step, previous_step)
            self.assertEqual(self.env.market_state.all(), self.env.market_data.iloc[previous_step].all())
      
    def test_recalculate_market_data(self):
        # Setup a mock INDICATOR dictionary
        mock_indicator = {
            'func': lambda df, window: df.rolling(window).mean(),
            'params': ['window']
        }

        # Assume that self.env is an instance of your trading environment and it has a mock original_market_data
        self.env.original_market_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Assume initial setting
        self.env.chosen_indicator = {
            'moving_average': {'window': 3}
        }
        
        self.env.INDICATORS = {
            'moving_average': mock_indicator
        }

        # Call recalculate_market_data function
        updated_data = self.env.recalculate_market_data()

        # Verify that market_data has been updated correctly
        # For moving_average: window 3
        expected_data = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pd.testing.assert_series_equal(updated_data['moving_average'], expected_data, check_dtype=False)
    
    def test_update_metrics(self):
        # Assume that self.env is an instance of your trading environment, 
        # and it has mocked methods for calculate_portfolio_value and calculate_drawdown
        self.env.current_step = 10
        self.env.performance_metrics = {'Running Average Value': 2000}
        self.env.winning_trades = 10
        self.env.total_trades = 20

        # Assume the mocked return values for the functions
        mock_portfolio_value = 3000
        mock_drawdown = 500

        # Mock the methods
        self.env.calculate_portfolio_value = unittest.mock.MagicMock(return_value=mock_portfolio_value)
        self.env.calculate_drawdown = unittest.mock.MagicMock(return_value=mock_drawdown)

        # Call update_metrics function
        self.env.update_metrics()

        # Verify the updated metrics
        expected_metrics = {
            'Portfolio Value': mock_portfolio_value,
            'Running Average Value': (self.env.performance_metrics['Running Average Value'] * self.env.current_step + mock_portfolio_value) / (self.env.current_step + 1),
            'Drawdown': mock_drawdown,
            'Winning Trades': self.env.winning_trades,
            'Total Trades': self.env.total_trades
        }
        assert self.env.performance_metrics == expected_metrics
        
    def test_calculate_portfolio_value(self):
        # Assume that self.env is an instance of your trading environment 
        # and it has a mock market_state and num_shares
        self.env.market_state = {'close': 150}
        self.env.num_shares = 10

        # Call calculate_portfolio_value function
        portfolio_value = self.env.calculate_portfolio_value()

        # Verify the calculated portfolio value
        self.assertEqual(portfolio_value, 150 * 10)
        
    def test_calculate_drawdown(self):
        # Assume that self.env is an instance of your trading environment

        # Case 1: historical_peaks doesn't exist
        drawdown = self.env.calculate_drawdown(2000)
        self.assertEqual(drawdown, 0)
        self.assertEqual(self.env.historical_peaks, 2000)

        # Case 2: current value is higher
        drawdown = self.env.calculate_drawdown(3000)
        self.assertEqual(drawdown, 0)
        self.assertEqual(self.env.historical_peaks, 3000)

        # Case 3: current value is lower
        drawdown = self.env.calculate_drawdown(2500)
        self.assertEqual(drawdown, (3000 - 2500) / 3000)
        self.assertEqual(self.env.historical_peaks, 3000)
    
    def test_calculate_total_trades(self):
        # Assume that self.env is an instance of your trading environment

        # Set the total trades count
        self.env.total_trades = 7

        # Call calculate_total_trades function
        total_trades = self.env.calculate_total_trades()

        # Verify the returned total trades count
        assert total_trades == 7
    
    def test_update_indicator_settings(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnvironment()

        # Test case: known indicator with adjustable parameters
        new_sma_settings = {'period': 50}
        trading_env.update_indicator_settings('sma', new_sma_settings)
        assert trading_env.chosen_indicators['sma'] == new_sma_settings

        # Test case: known indicator without adjustable parameters
        trading_env.update_indicator_settings('trange', {})
        assert trading_env.chosen_indicators['trange'] == {}

        # Test case: unrecognized indicator
        with pytest.raises(ValueError):
            trading_env.update_indicator_settings('unknown_indicator', {'some_setting': 10})
    
    def test_indicator_settings_to_vector(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnvironment()

        # Test case: known indicator settings
        indicator_settings = {'sma': 50, 'ema': 30, 'rsi': 14}
        expected_vector = np.array([50, 30, 14])
        assert np.array_equal(trading_env.indicator_settings_to_vector(indicator_settings), expected_vector)
    
    def test_calculate_indicators(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnvironment()

        # Define a known set of indicator settings
        trading_env.chosen_indicators = {
            'sma': {'period': 10},
            'rsi': {'period': 14}
        }

        # Assume the market_data is a pandas DataFrame with columns: 'Open', 'High', 'Low', 'Close', 'Volume'
        trading_env.market_data = pd.DataFrame(np.random.rand(100, 5), columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Call the calculate_indicators function
        trading_env.calculate_indicators()

        # Verify that the calculated indicators are added to the market data
        assert 'sma' in trading_env.market_data.columns
        assert 'rsi' in trading_env.market_data.columns     
         
    def test_select_indicators(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnvironment()

        # Define a known set of indicators
        chosen_indicators = ['sma', 'rsi']

        # Call the select_indicators function
        trading_env.select_indicators(chosen_indicators)

        # Verify that the selected indicators are updated in the environment
        assert set(trading_env.chosen_indicators.keys()) == set(chosen_indicators)

        # Test case: unrecognized indicator
        with pytest.raises(ValueError):
            trading_env.select_indicators(['unknown_indicator'])
    
    @pytest.mark.parametrize("total_rewards, expected_total_reward", [
    ([1, 1, 1, 1, 1], 5),
    ([10, -3, 2, 5], 14),
    ([-5, -5, -10], -20),
    ])
    def test_run_episode(total_rewards, expected_total_reward):
        # Initialize TradingEnv
        trading_env = TradingEnvironment()

        # Create a mock for the agent
        mock_agent = mock.Mock()
        mock_agent.get_action = mock.Mock()
        mock_agent.learn = mock.Mock()

        # Create a mock for the environment
        mock_env = mock.Mock()
        mock_env.reset = mock.Mock()
        mock_env.step = mock.Mock(side_effect=list(zip(
            ['state{}'.format(i) for i in range(len(total_rewards))], 
            total_rewards, 
            [False]*(len(total_rewards) - 1) + [True],  # Done is True only for the last step
            [None]*len(total_rewards)
        )))

        trading_env.agent = mock_agent
        trading_env.env = mock_env

        # Initialize state, balance, step, and sequence_length
        trading_env.state = 'initial_state'
        trading_env.balance = 1000
        trading_env.step = 0
        trading_env.sequence_length = 10
        trading_env.data = [None] * (len(total_rewards) + trading_env.sequence_length)

        # Run episode
        total_reward = trading_env.run_episode()

        assert total_reward == expected_total_reward
             
    def test_buy_asset(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnvironment()

        # Initialize values
        trading_env.cash_balance = 1000
        trading_env.market_state = {'close': 10}
        trading_env.num_shares = 0
        trading_env.total_trades = 0

        # Call buy_asset function
        trading_env.buy_asset(50)  # 50% of cash balance

        # Verify that the cash balance, number of shares and total trades are updated correctly
        assert trading_env.cash_balance == 500  # 1000 - 1000*50/100
        assert trading_env.num_shares == 50  # 1000*50/100 / 10
        assert trading_env.total_trades == 1

        # Call buy_asset function with cost more than available balance
        trading_env.buy_asset(200)  # 200% of cash balance

        # Verify that the cash balance, number of shares and total trades are not updated
        assert trading_env.cash_balance == 500  # remains same
        assert trading_env.num_shares == 50  # remains same
        assert trading_env.total_trades == 1  # remains same
        
    def test_sell_asset(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnvironment()

        # Initialize values
        trading_env.cash_balance = 1000
        trading_env.market_state = {'close': 10}
        trading_env.num_shares = 100
        trading_env.total_trades = 0
        trading_env.winning_trades = 0
        trading_env.buy_price = 9

        # Call sell_asset function
        trading_env.sell_asset(50)  # 50% of current shares

        # Verify that the cash balance, number of shares, winning trades and total trades are updated correctly
        assert trading_env.cash_balance == 1500  # 1000 + 100*50/100*10
        assert trading_env.num_shares == 50  # 100 - 100*50/100
        assert trading_env.total_trades == 1
        assert trading_env.winning_trades == 1  # selling price > buying price

        # Call sell_asset function with amount more than available shares
        trading_env.sell_asset(200)  # 200% of current shares

        # Verify that the cash balance, number of shares, winning trades and total trades are not updated
        assert trading_env.cash_balance == 1500  # remains same
        assert trading_env.num_shares == 50  # remains same
        assert trading_env.total_trades == 1  # remains same
        assert trading_env.winning_trades == 1  # remains same

if __name__ == '__main__':
    unittest.main()