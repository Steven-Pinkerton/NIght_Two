import unittest
from collections import defaultdict
from unittest.mock import patch
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
        self.assertEqual(self.env.portfolio, defaultdict(float))
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

        # Check that max_action is None
        self.assertIsNone(self.env.max_action)

    @patch('pandas.DataFrame', autospec=True)
    def test_calculate_max_action(self, mock_df):
        # Mock the market_data attribute
        self.env.market_data = mock_df
        # Note: we should set the return_value of mock_df.columns to return a list
        mock_df.columns.__iter__.return_value = ["AAPL", "GOOG", "TSLA"]  # Let's assume we have 3 stocks for simplicity

        # Here, we can create some test scenarios to validate the max_action calculation
        expected_max_action = 3 * 4 * 10  # 3 stocks * 4 actions * 10 percentages
        actual_max_action = self.env.calculate_max_action()

        self.assertEqual(actual_max_action, expected_max_action)
        
    def test_initialize_state(self):
        # Mock market data
        self.env.market_data = pd.DataFrame({
            'AAPL': [150, 151, 152],
            'GOOG': [2700, 2701, 2702],
            'TSLA': [600, 601, 602]
        })
        self.env.current_step = 0
        self.env.initial_cash_balance = 10000.0

        # Mock all indicators
        self.env.all_indicators = {'sma': {'period': 30}, 'rsi': {'period': 14}}

        # Call initialize_state
        returned_state = self.env.initialize_state()

        # Check the initial state of the environment
        self.assertEqual(self.env.portfolio, {})
        self.assertEqual(self.env.cash_balance, 10000.0)
        self.assertEqual(self.env.market_state.tolist(), [150, 2700, 600])
        self.assertIsNot(self.env.chosen_indicators, self.env.all_indicators)
        self.assertEqual(self.env.previous_action, None)
        self.assertEqual(self.env.buy_prices, {})
        self.assertEqual(self.env.sell_prices, {})
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
            np.array([30, 14]),  # chosen indicators
            np.array([0])  # previous action
        ])

        np.testing.assert_array_equal(self.env.state, expected_state)
        np.testing.assert_array_equal(returned_state, expected_state)

    def test_concatenate_state(self):
        # Mock portfolio
        self.env.portfolio = {'AAPL': 10, 'GOOG': 5}

        # Mock cash balance
        self.env.cash_balance = 5000.0

        # Mock market state
        self.env.market_state = pd.Series([150, 2700, 600], index=['AAPL', 'GOOG', 'TSLA'])

        # Mock performance metrics
        self.env.performance_metrics = {
            'Portfolio Value': 10000.0,
            'Running Average Value': 9500.0,
            'Drawdown': 500.0,
            'Winning Trades': 20,
            'Total Trades': 30
        }

        # Mock chosen indicators
        self.env.chosen_indicators = {'sma': 30, 'rsi': 14}

        # Mock previous action
        self.env.previous_action = 1

        # Call concatenate_state
        state = self.env.concatenate_state()

        # The state vector should be a concatenation of the portfolio vector, cash balance, market state, 
        # performance metrics, chosen indicators, and previous action. Each of these elements should be converted 
        # to the correct format before concatenation.

        expected_state = np.concatenate([
            np.array([10, 5, 0]),  # portfolio vector
            np.array([5000.0]),  # cash balance
            np.array([150, 2700, 600]),  # market state
            np.array([10000.0, 9500.0, 500.0, 20, 30]),  # performance metrics
            np.array([30, 14]),  # chosen indicators
            np.array([1])  # previous action
        ])

        # Check the output state
        np.testing.assert_array_equal(state, expected_state)
    
    def test_portfolio_to_vector(self):
        # Define test scenarios
        test_scenarios = [
            ({'AAPL': 10, 'GOOG': 5}, ['AAPL', 'GOOG', 'TSLA'], np.array([10, 5, 0])),
            ({}, ['AAPL', 'GOOG', 'TSLA'], np.array([0, 0, 0])),
            ({'AAPL': 5, 'GOOG': 5, 'TSLA': 5}, ['AAPL', 'GOOG', 'TSLA'], np.array([5, 5, 5])),
            ({'AAPL': 10, 'TSLA': 5}, ['AAPL', 'GOOG', 'TSLA'], np.array([10, 0, 5]))
        ]

        # Run each test scenario
        for portfolio, market_data_cols, expected_vector in test_scenarios:
            # Mock portfolio and market data
            self.env.portfolio = portfolio
            self.env.market_data = pd.DataFrame(columns=market_data_cols)

            # Call portfolio_to_vector
            vector = self.env.portfolio_to_vector()

            # Check the output vector
            np.testing.assert_array_equal(vector, expected_vector)
             
    def test_metrics_to_vector(self):
        # Mock metrics
        metrics = {
            'Portfolio Value': 10000.0,
            'Running Average Value': 9500.0,
            'Drawdown': 500.0,
            'Winning Trades': 20,
            'Total Trades': 30
        }

        # Call metrics_to_vector
        vector = self.env.metrics_to_vector(metrics)

        # The output should be an array with the metric values, in the order they appear in the metrics dictionary
        expected_vector = np.array([10000.0, 9500.0, 500.0, 20, 30])

        # Check the output vector
        np.testing.assert_array_equal(vector, expected_vector)
        
    def test_action_to_vector(self):
        # Define test scenarios
        test_scenarios = [
            (1, 1),  # Non-zero action
            (None, 0),  # None action
            (0, 0),  # Zero action
            (10, 10),  # Other non-zero action
        ]

        # Run each test scenario
        for action, expected_vector in test_scenarios:
            # Call action_to_vector
            vector = self.env.action_to_vector(action)

            # Check the output vector
            self.assertEqual(vector, expected_vector)
        
    def test_define_action_space(self):
            # Here, we can verify that the action space is correctly defined
            action_space = self.env.define_action_space()
            self.assertEqual(action_space, ['Buy', 'Sell', 'Hold', 'Change Settings'])
            self.assertEqual(self.env.action_space, action_space)

    def test_step_buy_action(self):
        # Mock action: buy 10 shares of AAPL
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 10}

        # Mock market state: AAPL price is $150
        self.env.market_state = pd.Series({'AAPL': 150.0})

        # Mock balance: $5000
        self.env.cash_balance = 5000.0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the portfolio was updated
        self.assertEqual(self.env.portfolio.get('AAPL'), 10)

        # Verify that the balance was updated
        self.assertEqual(self.env.cash_balance, 5000 - 150 * 10)

        # Verify that the reward was calculated
        # This check will depend on the specifics of the reward calculation function
        new_state, reward, done = self.env.step(action)
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)
    
    def test_step_sell_action(self):
        # Mock action: sell 5 shares of AAPL
        action = {'symbol': 'AAPL', 'type': 'sell', 'amount': 5}

        # Mock portfolio: 10 shares of AAPL
        self.env.portfolio = {'AAPL': 10}

        # Mock market state: AAPL price is $200
        self.env.market_state = pd.Series({'AAPL': 200.0})

        # Mock balance: $5000
        self.env.cash_balance = 5000.0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the portfolio was updated
        self.assertEqual(self.env.portfolio.get('AAPL'), 5)

        # Verify that the balance was updated
        self.assertEqual(self.env.cash_balance, 5000 + 200 * 5)

        # Verify that the reward was calculated This check will depend on the specifics of the reward calculation function
        # ...calculate expected reward based on your specific reward calculation
        new_state, reward, done = self.env.step(action)
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)
        
    def test_update_metrics_in_step_function():
        env = Environment(market_data, cash_balance=1000)

        # Buy action for the first symbol in market data
        action = {'symbol': env.market_data.columns[0], 'type': 'buy', 'amount': 10}
        previous_portfolio_value = env.current_portfolio_value

        # Step the environment
        state, reward, done = env.step(action)

        current_portfolio_value = env.calculate_portfolio_value()
        expected_running_average_value = (env.performance_metrics['Running Average Value'] * (env.current_step - 1) + current_portfolio_value) / env.current_step

        # Check that metrics are updated correctly
        assert env.performance_metrics['Portfolio Value'] == current_portfolio_value
        assert env.performance_metrics['Running Average Value'] == expected_running_average_value
        assert env.performance_metrics['Drawdown'] == env.calculate_drawdown(current_portfolio_value)
        assert env.performance_metrics['Winning Trades'] == env.calculate_winning_trades()
        assert env.performance_metrics['Total Trades'] == env.calculate_total_trades()
        
        # Check that portfolio value increased when a buy action was made
        assert env.current_portfolio_value > previous_portfolio_value

    def test_episode_end_with_all_steps_taken(self):
        # Mock action: buy 1 share of AAPL
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 1}

        # Mock portfolio: 0 shares of AAPL
        self.env.portfolio = {'AAPL': 0}

        # Mock market state: AAPL price is $200, only one row of data
        self.env.market_state = pd.Series({'AAPL': 200.0})
        self.env.market_data = pd.DataFrame([self.env.market_state])

        # Mock balance: $5000
        self.env.cash_balance = 5000.0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the episode has ended
        self.assertTrue(done)
        
    def test_episode_end_with_no_balance(self):
        # Mock action: buy 25 shares of AAPL
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 25}

        # Mock portfolio: 0 shares of AAPL
        self.env.portfolio = {'AAPL': 0}

        # Mock market state: AAPL price is $200
        self.env.market_state = pd.Series({'AAPL': 200.0})

        # Mock balance: $5000 (not enough to buy 25 shares)
        self.env.cash_balance = 5000.0

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the episode has ended
        self.assertTrue(done)
        
    def test_step_change_indicator_settings(self):
        # Mock action: change indicator settings
        action = {
            'change_indicator_settings': {
                'indicator_name': 'sma', 
                'settings': {'parameter1': 40}
            }
        }

        # Initial indicator settings
        self.env.indicator_settings = {}

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the indicator settings were updated (CHANGE THE PARAMETERS ETC TO SOMETHING WE ACTUALLY HAVE)
        expected_settings = {'my_indicator': {'parameter1': 10, 'parameter2': 5}}
        self.assertEqual(self.env.indicator_settings, expected_settings)
        
    def test_step_previous_action(self):
        # Mock action: buy 5 shares of AAPL
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 5}

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the previous action was stored correctly
        self.assertEqual(self.env.previous_action, action)
        
    def test_step_state_updates_portfolio(self):
        # ...setup for action...

        # Copy the current portfolio_vector before the action
        previous_portfolio_vector = self.env.portfolio_to_vector().copy()

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the portfolio_vector was updated
        self.assertNotEqual(self.env.portfolio_to_vector(), previous_portfolio_vector)

    def test_step_updates_cash_balance_vector(self):
        # Mock action: buy 5 shares of AAPL
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 5}

        # Get the cash_balance_vector before the action
        previous_cash_balance_vector = np.array([self.env.cash_balance])

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the cash_balance_vector was updated (the cash balance should be less now)
        self.assertTrue(np.array([self.env.cash_balance]).all() < previous_cash_balance_vector.all())

    def test_step_updates_performance_vector(self):
        # Mock action: buy 5 shares of AAPL
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 5}

        # Get the performance_vector before the action
        previous_performance_vector = self.env.metrics_to_vector(self.env.performance_metrics)

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the performance_vector was updated
        self.assertFalse((self.env.metrics_to_vector(self.env.performance_metrics) == previous_performance_vector).all())

    def test_step_updates_indicator_vector(self):
        # Mock action: change indicator settings
        action = {'change_indicator_settings': {'sma': {'period': 40}}}

        # Get the indicator_vector before the action
        previous_indicator_vector = self.env.indicator_settings_to_vector(self.env.chosen_indicators)

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the indicator_vector was updated
        self.assertFalse((self.env.indicator_settings_to_vector(self.env.chosen_indicators) == previous_indicator_vector).all())

    def test_step_updates_action_vector(self):
        # Mock action: buy 5 shares of AAPL
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 5}

        # Get the action_vector before the action
        previous_action_vector = self.env.action_to_vector(self.env.previous_action)

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the action_vector was updated
        self.assertNotEqual(self.env.action_to_vector(self.env.previous_action), previous_action_vector)

    def test_step_state_updates_cash_balance(self):
        # ...setup for action...

        # Copy the current cash_balance_vector before the action
        previous_cash_balance_vector = np.array([self.env.cash_balance])

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the cash_balance_vector was updated
        self.assertNotEqual(np.array([self.env.cash_balance]), previous_cash_balance_vector)

    def test_step_state_updates_performance_metrics(self):
        # ...setup for action...
        
        # Copy the current performance_vector before the action
        previous_performance_vector = self.env.metrics_to_vector(self.env.performance_metrics).copy()
        
        # Call the step function
        new_state, reward, done = self.env.step(action)
        
        
        # Verify that the performance_vector was updated
        self.assertNotEqual(self.env.metrics_to_vector(self.env.performance_metrics), previous_performance_vector)
            
    def test_step_state_updates_indicator_vector(self):
        # ...setup for action...
        
        # Copy the current indicator_vector before the action
        previous_indicator_vector = self.env.indicator_to_vector().copy()
        
        # Call the step function
        new_state, reward, done = self.env.step(action)
        
        # Verify that the indicator_vector was updated
        self.assertNotEqual(self.env.indicator_to_vector(), previous_indicator_vector)

    def test_step_state_updates_action_vector(self):
        # ...setup for action...
        
        # Copy the current action_vector before the action
        previous_action_vector = self.env.action_to_vector(self.env.previous_action).copy()
        
        # Call the step function
        new_state, reward, done = self.env.step(action)
        
        # Verify that the action_vector was updated
        self.assertNotEqual(self.env.action_to_vector(self.env.previous_action), previous_action_vector)

    def test_step_state_concatenation(self):
        # ...setup for action...

        # Copy the current state before the action
        previous_state = self.env.state.copy()

        # Call the step function
        new_state, reward, done = self.env.step(action)

        # Verify that the state was updated
        self.assertNotEqual(self.env.state, previous_state)

        # Verify that the state includes all components
        self.assertEqual(len(self.env.state), len(self.env.portfolio_to_vector()) + 1 + len(self.env.market_state.values) + len(self.env.metrics_to_vector(self.env.performance_metrics)) + len(self.env.indicator_settings_to_vector(self.env.chosen_indicators)) + len(self.env.action_to_vector(self.env.previous_action)))

    def test_valid_buy_action_updates_portfolio(self):
        self.env.cash_balance = 2000.0
        action = {"symbol": "AAPL", "type": "buy", "amount": 10}

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.portfolio["AAPL"], 10)
        self.assertEqual(self.env.cash_balance, 500.0)

    def test_valid_sell_action_updates_portfolio(self):
        self.env.cash_balance = 2000.0
        self.env.portfolio["AAPL"] = 10
        self.env.buy_prices["AAPL"] = [100.0] * 10  # Mock buy prices
        action = {"symbol": "AAPL", "type": "sell", "amount": 5}

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.portfolio["AAPL"], 5)
        self.assertEqual(self.env.cash_balance, 2750.0)

    def test_insufficient_cash_prevents_buy(self):
        self.env.cash_balance = 1000.0
        action = {"symbol": "AAPL", "type": "buy", "amount": 10}

        self.env.update_portfolio_and_balance(action)

        self.assertNotIn("AAPL", self.env.portfolio)
        self.assertEqual(self.env.cash_balance, 1000.0)

    def test_insufficient_shares_prevents_sell(self):
        self.env.cash_balance = 2000.0
        self.env.portfolio["AAPL"] = 3
        action = {"symbol": "AAPL", "type": "sell", "amount": 5}

        self.env.update_portfolio_and_balance(action)

        self.assertEqual(self.env.portfolio["AAPL"], 3)
        self.assertEqual(self.env.cash_balance, 2000.0)

    def test_invalid_action_does_nothing(self):
        self.env.cash_balance = 2000.0
        action = {"symbol": "AAPL", "type": "fly", "amount": 10}  # Invalid action type

        self.env.update_portfolio_and_balance(action)

        self.assertNotIn("AAPL", self.env.portfolio)
        self.assertEqual(self.env.cash_balance, 2000.0)
      
    def test_calculate_initial_metrics(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnv()

        # Call the calculate_initial_metrics function
        trading_env.calculate_initial_metrics()

        # Verify the calculated initial metrics
        self.assertEqual(trading_env.performance_metrics['Portfolio Value'], trading_env.calculate_portfolio_value())
        self.assertEqual(trading_env.performance_metrics['Running Average Value'], trading_env.calculate_portfolio_value())
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
         
        else:
            # If the previous step was not less than the length of the market data minus one,
            # the current step should still be the same
            self.assertEqual(self.env.current_step, previous_step)
        
    def test_is_valid_action(self):
        # Assuming self.env is an instance of your trading environment,
        # and it has been initialized with some mock market data.

        # Invalid action: not a dictionary
        action = "Buy 5 AAPL"
        self.assertFalse(self.env.is_valid_action(action))

        # Invalid action: missing keys
        action = {'symbol': 'AAPL', 'type': 'buy'}
        self.assertFalse(self.env.is_valid_action(action))

        # Invalid action: action type not in the action space
        action = {'symbol': 'AAPL', 'type': 'borrow', 'amount': 5}
        self.assertFalse(self.env.is_valid_action(action))

        # Invalid action: action symbol not in the market state
        action = {'symbol': 'XYZ', 'type': 'buy', 'amount': 5}
        self.assertFalse(self.env.is_valid_action(action))

        # Invalid action: action amount is negative
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': -5}
        self.assertFalse(self.env.is_valid_action(action))

        # Invalid action: action amount is not a number
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 'five'}
        self.assertFalse(self.env.is_valid_action(action))

        # Valid action
        action = {'symbol': 'AAPL', 'type': 'buy', 'amount': 5}
        self.assertTrue(self.env.is_valid_action(action))
      
    def test_recalculate_market_data(self):
        # Setup a mock INDICATORS dictionary
        mock_indicators = {
            'moving_average': {
                'func': lambda df, window: df.rolling(window).mean(),
                'params': ['window']
            },
            'exponential_moving_average': {
                'func': lambda df, span: df.ewm(span=span).mean(),
                'params': ['span']
            }
        }
        # Assume that self.env is an instance of your trading environment and it has a mock original_market_data and mock_indicators
        self.env.original_market_data = pd.DataFrame({
            'AAPL': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'GOOGL': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })

        # Assume initial settings
        self.env.indicator_settings = {
            'moving_average': {'window': 3},
            'exponential_moving_average': {'span': 3}
        }

        # Call recalculate_market_data function
        updated_data = self.env.recalculate_market_data()

        # Verify that market_data has been updated correctly
        # For moving_average: window 3
        expected_data = pd.DataFrame({
            'AAPL': [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'GOOGL': [np.nan, np.nan, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
        })
        pd.testing.assert_frame_equal(updated_data['moving_average'], expected_data, check_dtype=False)

        # Verify that market_data has been updated correctly
        # For exponential_moving_average: span 3
        expected_data = pd.DataFrame({
            'AAPL': [1.0, 1.6667, 2.4286, 3.2667, 4.1613, 5.0968, 6.0645, 7.0591, 8.0755, 9.0995],
            'GOOGL': [10.0, 9.3333, 8.5714, 7.7333, 6.8387, 5.9032, 4.9355, 3.9409, 2.9245, 1.9005]
        })
        pd.testing.assert_frame_equal(updated_data['exponential_moving_average'].round(4), expected_data, check_dtype=False)
    
    def test_update_metrics(self):
        # Assume that self.env is an instance of your trading environment, and it has mocked methods for calculate_portfolio_value, calculate_drawdown, calculate_winning_trades, and calculate_total_trades
        self.env.current_step = 10
        self.env.performance_metrics = {'Running Average Value': 2000}

        # Assume the mocked return values for the functions
        mock_portfolio_value = 3000
        mock_drawdown = 500
        mock_winning_trades = 10
        mock_total_trades = 20

        # Mock the methods
        self.env.calculate_portfolio_value = MagicMock(return_value=mock_portfolio_value)
        self.env.calculate_drawdown = MagicMock(return_value=mock_drawdown)
        self.env.calculate_winning_trades = MagicMock(return_value=mock_winning_trades)
        self.env.calculate_total_trades = MagicMock(return_value=mock_total_trades)

        # Call update_metrics function
        self.env.update_metrics()

        # Verify the updated metrics
        expected_metrics = {
            'Portfolio Value': mock_portfolio_value,
            'Running Average Value': (self.env.performance_metrics['Running Average Value'] * (self.env.current_step - 1) + mock_portfolio_value) / self.env.current_step,
            'Drawdown': mock_drawdown,
            'Winning Trades': mock_winning_trades,
            'Total Trades': mock_total_trades
        }
        assert self.env.performance_metrics == expected_metrics
    
    def test_calculate_portfolio_value(self):
        # Assume that self.env is an instance of your trading environment and it has a mock market_state and portfolio
        self.env.market_state = {'AAPL': 150, 'GOOGL': 2500}
        self.env.portfolio = {'AAPL': 10, 'GOOGL': 1}

        # Call calculate_portfolio_value function
        portfolio_value = self.env.calculate_portfolio_value()

        # Verify the calculated portfolio value
        assert portfolio_value == 150*10 + 2500*1
        
    def test_calculate_drawdown(self):
        # Assume that self.env is an instance of your trading environment and it has a mock calculate_portfolio_value method
        # We'll simulate three scenarios: when historical_peaks attribute doesn't exist, when the current value is higher, and when it's lower

        # Case 1: historical_peaks doesn't exist
        self.env.calculate_portfolio_value = MagicMock(return_value=2000)
        drawdown = self.env.calculate_drawdown()
        assert drawdown == 0
        assert self.env.historical_peaks == 2000

        # Case 2: current value is higher
        self.env.calculate_portfolio_value = MagicMock(return_value=3000)
        drawdown = self.env.calculate_drawdown()
        assert drawdown == 0
        assert self.env.historical_peaks == 3000

        # Case 3: current value is lower
        self.env.calculate_portfolio_value = MagicMock(return_value=2500)
        drawdown = self.env.calculate_drawdown()
        assert drawdown == (3000 - 2500) / 3000
        assert self.env.historical_peaks == 3000 
        
    def test_calculate_winning_trades(self):
        # Assume that self.env is an instance of your trading environment and it has winning_trades attribute

        # Set the winning trades count
        self.env.winning_trades = 5

        # Call calculate_winning_trades function
        winning_trades = self.env.calculate_winning_trades()

        # Verify the returned winning trades count
        assert winning_trades == 5
    
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
        trading_env = TradingEnv()

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
        trading_env = TradingEnv()

        # Test case: known indicator settings
        indicator_settings = {'sma': 50, 'ema': 30, 'rsi': 14}
        expected_vector = np.array([50, 30, 14])
        assert np.array_equal(trading_env.indicator_settings_to_vector(indicator_settings), expected_vector)
    
    def test_calculate_indicators(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnv()

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
        
    @pytest.mark.parametrize("total_rewards, expected_total_reward", [
    ([1, 1, 1, 1, 1], 5),
    ([10, -3, 2, 5], 14),
    ([-5, -5, -10], -20),
    ])
    def test_run_episode(total_rewards, expected_total_reward):
        # Initialize TradingEnv
        trading_env = TradingEnv()

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

if __name__ == '__main__':
    unittest.main()