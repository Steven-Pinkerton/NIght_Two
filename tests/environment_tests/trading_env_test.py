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
        self.env = TradingEnvironment()
        self.env.historical_peaks = 10
            
    def test_init(self):
        # Test that the environment is initialized correctly
        self.assertEqual(self.env.cash_balance, 10000.0)
        self.assertEqual(self.env.transaction_cost, 0.01)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.num_shares, 0)
        self.assertEqual(self.env.initial_cash_balance, 10000.0)

        # Check that all indicators have been initialized
        self.assertEqual(set(self.env.indicator_values.keys()), set(self.env.INDICATORS.keys()))
        self.assertEqual(set(self.env.params_values.keys()), set(self.env.INDICATORS.keys()))
        # All indicator values should be None at this point
        self.assertTrue(all(value is None for value in self.env.indicator_values.values()))

        # Check that chosen_indicators is an empty dictionary
        self.assertEqual(self.env.chosen_indicators, {})

        # Check that risk_adjusted_return_history and portfolio_value_history are empty lists
        self.assertEqual(self.env.risk_adjusted_return_history, [])
        self.assertEqual(self.env.portfolio_value_history, [])

        # Check that max_action has been calculated and is not None
        self.assertIsNotNone(self.env.max_action)
           
    def test_step_change_indicator_settings(self):
        from pandas.testing import assert_frame_equal
        # Mocking the `random.choice` function to control the output during testing
        with mock.patch('random.choice', return_value=20):
            # Creating a `TradingEnvironment` instance
            trading_env = TradingEnvironment()

            # Set the current step to a specific point
            trading_env.current_step = 10

            # Set a test indicator
            trading_env.chosen_indicators = {
                'sma': {'period': range(10, 51)}
            }

            # Setting up the new settings
            new_settings = {
                'sma': {'period': 20}
            }

            # Setting up the action
            action = {
                'type': 'change_indicator_settings',
                'settings': new_settings
            }

            with mock.patch.object(trading_env, 'calculate_indicators', return_value=None) as mock_calculate_indicators:
                # Execute step function with 'change_indicator_settings' action
                state, reward, done = trading_env.step(action)

                # Assert that `calculate_indicators` was called
                mock_calculate_indicators.assert_called_once()

            # Assert that new settings are applied correctly
            assert trading_env.chosen_indicators['sma']['period'] == 20

            # Assert that market data is correctly updated
            assert 'sma' in trading_env.market_data.columns

            # Assert exception when update_indicator_settings is called with invalid settings
            with pytest.raises(ValueError):
                trading_env.update_indicator_settings({'invalid_indicator': {'period': 20}})

            # Assert recalculate_market_data correctly resets the market_data
            original_columns = trading_env.original_market_data.columns
            assert_frame_equal(trading_env.market_data[original_columns], trading_env.original_market_data)

            # Assert calculate_indicators correctly handles parameter ranges
            trading_env.chosen_indicators = {
                'sma': {'period': range(10, 51)}
            }
            trading_env.calculate_indicators()
            assert trading_env.actual_params_values['sma']['period'] == 20

                        
    def test_select_indicators_updates_chosen_indicators_correctly(self):
        # Given
        agent = TradingEnvironment()
        action = {
            'type': 'select_indicators',
            'indicators': ['sma', 'ema']
        }

        # When
        state, reward, done = agent.step(action)

        # Then
        for indicator_name in action['indicators']:
            assert indicator_name in agent.chosen_indicators
        
    def test_select_indicators_selects_random_initial_settings(self):
        # Given
        agent = TradingEnvironment()
        action = {
            'type': 'select_indicators',
            'indicators': ['sma', 'ema']
        }

        # When
        state, reward, done = agent.step(action)

        # Then
        for indicator_name in action['indicators']:
            assert 'period' in agent.chosen_indicators[indicator_name]
            assert min(agent.INDICATORS[indicator_name]['params']['period']) <= agent.chosen_indicators[indicator_name]['period'] < max(agent.INDICATORS[indicator_name]['params']['period'])
            
    def test_select_indicators_calls_recalculate_market_data(self):
        # Given
        agent = TradingEnvironment()
        action = {
            'type': 'select_indicators',
            'indicators': ['sma', 'ema']
        }

        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            'Close': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        # When
        with patch.object(agent, 'recalculate_market_data') as mock_recalculate:
            mock_recalculate.return_value = mock_df
            state, reward, done = agent.step(action)

        # Then
        mock_recalculate.assert_called_once()
        
    def test_select_indicators_raises_error_for_unknown_indicator(self):
        # Given
        agent = TradingEnvironment()
        action = {
            'type': 'select_indicators',
            'indicators': ['sma', 'unknown_indicator']
        }

        # When & Then
        with pytest.raises(ValueError, match='Unknown indicator: unknown_indicator'):
            state, reward, done = agent.step(action)
            
    def test_buy_asset_updates_cash_balance_and_num_shares_and_total_trades(self):
        # Given
        agent = TradingEnvironment()
        action = {
            'type': 'buy',
            'percentage': 10
        }
        initial_cash_balance = agent.cash_balance
        initial_num_shares = agent.num_shares
        initial_total_trades = agent.total_trades
        cost = initial_cash_balance * (action['percentage'] / 100)
        print(agent.market_state)
        amount = cost / agent.market_state['Close']

        # When
        state, reward, done = agent.step(action)

        # Then
        assert agent.cash_balance == initial_cash_balance - cost
        assert agent.num_shares == initial_num_shares + amount
        assert agent.total_trades == initial_total_trades + 1
        
    def test_sell_asset_updates_cash_balance_and_num_shares_and_total_trades_and_potentially_winning_trades(self):
        # Given
        agent = TradingEnvironment()
        action = {
            'type': 'sell',
            'percentage': 10
        }
        initial_cash_balance = agent.cash_balance
        initial_num_shares = agent.num_shares
        initial_total_trades = agent.total_trades
        initial_winning_trades = agent.winning_trades
        amount = initial_num_shares * (action['percentage'] / 100)
        proceeds = agent.market_state['Close'] * amount
        win_trade = agent.market_state['Close'] > agent.buy_price

        # When
        state, reward, done = agent.step(action)

        # Then
        assert agent.cash_balance == initial_cash_balance + proceeds
        assert agent.num_shares == initial_num_shares - amount
        assert agent.total_trades == initial_total_trades + 1
        if win_trade:
            assert agent.winning_trades == initial_winning_trades + 1
        else:
            assert agent.winning_trades == initial_winning_trades
            
    def test_calculate_portfolio_value(self):
        # Given
        agent = TradingEnvironment()
        close_price = agent.market_state['Close']
        num_shares = agent.num_shares

        # When
        calculated_value = agent.calculate_portfolio_value()

        # Then
        assert calculated_value == close_price * num_shares
        
    def test_update_metrics(self):
        # Given
        agent = TradingEnvironment()
        initial_metrics = agent.performance_metrics

        # When
        portfolio_value = 1000.0
        with patch.object(agent, 'calculate_portfolio_value', return_value=portfolio_value):
            agent.update_metrics()
            current_portfolio_value = agent.calculate_portfolio_value()

        # Then
        running_average_value = (initial_metrics['Running Average Value'] * (agent.current_step - 1) + current_portfolio_value) / agent.current_step
        drawdown = agent.calculate_drawdown(current_portfolio_value)
        assert agent.performance_metrics == {
            'Portfolio Value': current_portfolio_value,
            'Running Average Value': running_average_value,
            'Drawdown': drawdown,
            'Winning Trades': agent.winning_trades, 
            'Total Trades': agent.total_trades 
        }
            
    def test_concatenate_state(self):
        # Given
        agent = TradingEnvironment()

        # When
        full_state = agent.concatenate_state()

        # Then
        num_shares_vector = np.array([agent.num_shares])
        cash_balance_vector = np.array([agent.cash_balance])
        performance_vector = agent.metrics_to_vector(agent.performance_metrics)
        indicator_vector = agent.indicator_settings_to_vector(agent.chosen_indicators)
        expected_state = np.concatenate([num_shares_vector, cash_balance_vector, agent.market_state.values, performance_vector, indicator_vector])
        assert np.array_equal(full_state, expected_state)
    
    def test_calculate_reward(self):
        # Given
        agent = TradingEnvironment()
        agent.previous_portfolio_value = 100
        agent.current_portfolio_value = 120
        agent.transaction_cost = 0.01
        agent.portfolio_value_history = [100, 105, 110, 115, 120]  # Add more values here
        
        # When
        reward = agent.calculate_reward()

        # Then
        assert reward >= 0
        assert isinstance(reward, float)
        
if __name__ == '__main__':
    unittest.main()