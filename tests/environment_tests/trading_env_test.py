import unittest
import mock
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
           
    def test_step_change_indicator_settings():
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

            # Execute step function with 'change_indicator_settings' action
            state, reward, done = trading_env.step(action)

            # Assert that new settings are applied correctly
            assert trading_env.chosen_indicators['sma']['period'] == 20

            # Assert that market data is correctly updated
            assert 'sma' in trading_env.market_data.columns

            # Assert that calculate_indicators is called with the correct parameters
            calculate_indicators_params = trading_env.calculate_indicators.call_args[1]
            assert calculate_indicators_params['params_values'] == trading_env.params_values

            # Assert exception when update_indicator_settings is called with invalid settings
            with pytest.raises(ValueError):
                trading_env.update_indicator_settings({'invalid_indicator': {'period': 20}})

            # Assert recalculate_market_data correctly resets the market_data
            assert_frame_equal(trading_env.market_data, trading_env.original_market_data)

            # Assert calculate_indicators correctly handles parameter ranges
            trading_env.chosen_indicators = {
                'sma': {'period': range(10, 51)}
            }
            trading_env.calculate_indicators()
            assert trading_env.actual_params_values['sma']['period'] == 20
            
    def test_select_indicators_updates_chosen_indicators_correctly():
        # Given
        agent = TradingAgent()
        action = {
            'type': 'select_indicators',
            'indicators': ['SMA', 'EMA']
        }

        # When
        state, reward, done = agent.step(action)

        # Then
        for indicator_name in action['indicators']:
            assert indicator_name in agent.chosen_indicators
        
    def test_select_indicators_selects_random_initial_settings():
        # Given
        agent = TradingAgent()
        action = {
            'type': 'select_indicators',
            'indicators': ['SMA', 'EMA']
        }

        # When
        state, reward, done = agent.step(action)

        # Then
        for indicator_name in action['indicators']:
            assert 'period' in agent.chosen_indicators[indicator_name]
            assert agent.all_indicators[indicator_name].start <= agent.chosen_indicators[indicator_name]['period'] < agent.all_indicators[indicator_name].stop
            
    def test_select_indicators_calls_recalculate_market_data():
        # Given
        agent = TradingAgent()
        action = {
            'type': 'select_indicators',
            'indicators': ['SMA', 'EMA']
        }

        # When
        with patch.object(agent, 'recalculate_market_data') as mock_recalculate:
            state, reward, done = agent.step(action)

        # Then
        mock_recalculate.assert_called_once()
        
    def test_select_indicators_raises_error_for_unknown_indicator():
        # Given
        agent = TradingAgent()
        action = {
            'type': 'select_indicators',
            'indicators': ['SMA', 'unknown_indicator']
        }

        # When & Then
        with pytest.raises(ValueError, match='Unknown indicator: unknown_indicator'):
            state, reward, done = agent.step(action)
            
    def test_buy_asset_updates_cash_balance_and_num_shares_and_total_trades():
        # Given
        agent = TradingAgent()
        action = {
            'type': 'buy',
            'percentage': 10
        }
        initial_cash_balance = agent.cash_balance
        initial_num_shares = agent.num_shares
        initial_total_trades = agent.total_trades
        cost = initial_cash_balance * (action['percentage'] / 100)
        amount = cost / agent.market_state['close']

        # When
        state, reward, done = agent.step(action)

        # Then
        assert agent.cash_balance == initial_cash_balance - cost
        assert agent.num_shares == initial_num_shares + amount
        assert agent.total_trades == initial_total_trades + 1
        
    def test_sell_asset_updates_cash_balance_and_num_shares_and_total_trades_and_potentially_winning_trades():
        # Given
        agent = TradingAgent()
        action = {
            'type': 'sell',
            'percentage': 10
        }
        initial_cash_balance = agent.cash_balance
        initial_num_shares = agent.num_shares
        initial_total_trades = agent.total_trades
        initial_winning_trades = agent.winning_trades
        amount = initial_num_shares * (action['percentage'] / 100)
        proceeds = agent.market_state['close'] * amount
        win_trade = agent.market_state['close'] > agent.buy_price

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
            
    def test_calculate_portfolio_value():
        # Given
        agent = TradingAgent()
        close_price = agent.market_state['close']
        num_shares = agent.num_shares

        # When
        calculated_value = agent.calculate_portfolio_value()

        # Then
        assert calculated_value == close_price * num_shares
        
    def test_update_metrics():
        # Given
        agent = TradingAgent()
        initial_metrics = agent.performance_metrics

        # When
        agent.update_metrics()

        # Then
        current_portfolio_value = agent.calculate_portfolio_value()
        running_average_value = (initial_metrics['Running Average Value'] * (agent.current_step - 1) + current_portfolio_value) / agent.current_step
        drawdown = agent.calculate_drawdown(current_portfolio_value)
        assert agent.performance_metrics == {
            'Portfolio Value': current_portfolio_value,
            'Running Average Value': running_average_value,
            'Drawdown': drawdown,
            'Winning Trades': agent.winning_trades, 
            'Total Trades': agent.total_trades 
        }
        
    def test_concatenate_state():
        # Given
        agent = TradingAgent()

        # When
        full_state = agent.concatenate_state()

        # Then
        num_shares_vector = np.array([agent.num_shares])
        cash_balance_vector = np.array([agent.cash_balance])
        performance_vector = agent.metrics_to_vector(agent.performance_metrics)
        indicator_vector = agent.indicator_settings_to_vector(agent.chosen_indicators)
        action_vector = agent.action_to_vector(agent.previous_action)
        expected_state = np.concatenate([num_shares_vector, cash_balance_vector, agent.market_state.values, performance_vector, indicator_vector, action_vector])
        assert np.array_equal(full_state, expected_state)
    
    def test_calculate_reward():
        # Given
        agent = TradingAgent()
        initial_portfolio_value = agent.previous_portfolio_value

        # When
        reward = agent.calculate_reward()

        # Then
        ret = (agent.current_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        risk = np.std(agent.portfolio_value_history)
        EPSILON = 1e-8
        risk += EPSILON
        risk_adjusted_return = ret / risk
        trade_penalty = agent.transaction_cost * (agent.current_portfolio_value != agent.previous_portfolio_value)
        improvement_bonus = 0.0
        lookback_period = 10
        if agent.current_step > lookback_period:
            old_risk_adjusted_return = agent.risk_adjusted_return_history[-lookback_period]
            old_portfolio_value = agent.portfolio_value_history[-lookback_period]
            if risk_adjusted_return > old_risk_adjusted_return and agent.current_portfolio_value > old_portfolio_value:
                improvement_bonus = 0.1
        expected_reward = risk_adjusted_return - trade_penalty + improvement_bonus
        assert reward == expected_reward