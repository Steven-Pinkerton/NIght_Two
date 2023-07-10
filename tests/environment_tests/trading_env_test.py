import unittest
from collections import defaultdict
from night_two.environment.trading_env import TradingEnvironment  # Replace with the actual import

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

        # Continue with other initial properties

    def test_define_action_space(self):
            # Here, we can verify that the action space is correctly defined
            action_space = self.env.define_action_space()
            self.assertEqual(action_space, ['Buy', 'Sell', 'Hold', 'Change Settings'])
            self.assertEqual(self.env.action_space, action_space)

    @patch("pandas.DataFrame", autospec=True)  # Mocking pandas DataFrame
    def test_calculate_max_action(self, mock_df):
            # Mock the market_data attribute
            self.env.market_data = mock_df
            mock_df.columns = ["AAPL", "GOOG", "TSLA"]  # Let's assume we have 3 stocks for simplicity

            # Here, we can create some test scenarios to validate the max_action calculation
            expected_max_action = 3 * 4 * 10  # 3 stocks * 4 actions * 10 percentages
            actual_max_action = self.env.calculate_max_action()

            self.assertEqual(actual_max_action, expected_max_action)
            
    @patch("your_module.TradingEnvironment.calculate_indicators")
    @patch("data_preprocessing.preprocess_data")
    def test_load_market_data(self, mock_preprocess_data, mock_calculate_indicators):
        # Prepare a mock DataFrame
        mock_df = pd.DataFrame({
            'AAPL': [150, 151, 152],
            'GOOG': [2700, 2701, 2702],
            'TSLA': [600, 601, 602]
        })

        # Set the return value for preprocess_data
        mock_preprocess_data.return_value = mock_df

        # Call load_market_data
        self.env.load_market_data("dummy_source")

        # Verify preprocess_data was called with the correct argument
        mock_preprocess_data.assert_called_once_with("dummy_source")

        # Verify calculate_indicators was called with the correct argument
        mock_calculate_indicators.assert_called_once_with(self.env.all_indicators)

        # Verify the result of load_market_data is correct
        pd.testing.assert_frame_equal(self.env.market_data, mock_df)
          
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
        self.env.initialize_state()

        # Check the initial state of the environment
        self.assertEqual(self.env.portfolio, {})
        self.assertEqual(self.env.cash_balance, 10000.0)
        self.assertEqual(self.env.market_state.tolist(), [150, 2700, 600])
        self.assertEqual(self.env.chosen_indicators, self.env.all_indicators)
        self.assertEqual(self.env.previous_action, None)
        self.assertEqual(self.env.buy_prices, {})
        self.assertEqual(self.env.sell_prices, {})
        self.assertEqual(self.env.winning_trades, 0)
        self.assertEqual(self.env.total_trades, 0)

        # Check the state vector
        # The state vector should be a concatenation of the portfolio vector, cash balance, market state, 
        # performance metrics, chosen indicators, and previous action. Each of these elements should be converted 
        # to the correct format before concatenation.

        expected_state = np.concatenate([
            np.zeros(3),  # portfolio vector (empty portfolio)
            np.array([10000.0]),  # cash balance
            np.array([150, 2700, 600]),  # market state
            np.array([10000.0, 10000.0, 0, 0, 0]),  # performance metrics
            np.array([30, 14]),  # chosen indicators
            np.array([0])  # previous action
        ])

        np.testing.assert_array_equal(self.env.state, expected_state)

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
        # Mock portfolio
        self.env.portfolio = {'AAPL': 10, 'GOOG': 5}

        # Mock market data
        self.env.market_data = pd.DataFrame(columns=['AAPL', 'GOOG', 'TSLA'])

        # Call portfolio_to_vector
        vector = self.env.portfolio_to_vector()

        # The output should be an array with the number of shares held for each stock, in the order they appear in the market data
        expected_vector = np.array([10, 5, 0])

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
        # Test with a non-None action
        action = 1

        # Call action_to_vector
        vector = self.env.action_to_vector(action)

        # The output should be the action itself, as per the implementation
        self.assertEqual(vector, action)

        # Test with a None action
        action = None

        # Call action_to_vector
        vector = self.env.action_to_vector(action)

        # The output should be 0, representing "no action"
        self.assertEqual(vector, 0)

    def setUp(self):
        # Set up a new environment for each test
        self.env = TradingEnvironment()

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
                'indicator_name': 'my_indicator', 
                'settings': {'parameter1': 10, 'parameter2': 5}
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

    # Similar tests for cash_balance_vector, performance_vector, indicator_vector, action_vector...

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

    def test_calculate_initial_metrics(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnv()

        # Call the calculate_initial_metrics function
        trading_env.calculate_initial_metrics()

        # Verify the calculated initial metrics
        assert trading_env.performance_metrics['Portfolio Value'] == trading_env.calculate_portfolio_value()
        assert trading_env.performance_metrics['Running Average Value'] == trading_env.calculate_portfolio_value()
        assert trading_env.performance_metrics['Drawdown'] == 0
        assert trading_env.performance_metrics['Winning Trades'] == 0
        assert trading_env.performance_metrics['Total Trades'] == 0

    def test_calculate_portfolio_value(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnv()

        # Set the market state and portfolio manually
        trading_env.market_state = {'AAPL': 150.00, 'GOOG': 1000.00}
        trading_env.portfolio = {'AAPL': 10, 'GOOG': 2}

        # Call the calculate_portfolio_value function
        portfolio_value = trading_env.calculate_portfolio_value()

        # Verify the calculated portfolio value
        expected_value = (150.00 * 10) + (1000.00 * 2)
        assert portfolio_value == expected_value
        
    def test_update_metrics(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnv()

        # Set some initial values
        trading_env.current_step = 1
        trading_env.portfolio = {'AAPL': 10, 'GOOG': 2}
        trading_env.market_state = {'AAPL': 150.00, 'GOOG': 1000.00}
        trading_env.performance_metrics = {
            'Portfolio Value': 0,
            'Running Average Value': 0,
            'Drawdown': 0,
            'Winning Trades': 0,
            'Total Trades': 0
        }
        trading_env.winning_trades = 2
        trading_env.total_trades = 5

        # Call the update_metrics function
        trading_env.update_metrics()

        # Verify the updated metrics
        assert trading_env.performance_metrics['Portfolio Value'] == trading_env.calculate_portfolio_value()
        assert trading_env.performance_metrics['Running Average Value'] == trading_env.calculate_portfolio_value()  # Since it's the first step
        assert trading_env.performance_metrics['Drawdown'] == trading_env.calculate_drawdown()
        assert trading_env.performance_metrics['Winning Trades'] == trading_env.winning_trades
        assert trading_env.performance_metrics['Total Trades'] == trading_env.total_trades

    def test_calculate_drawdown(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnv()

        # Set some initial values
        trading_env.portfolio = {'AAPL': 10, 'GOOG': 2}
        trading_env.market_state = {'AAPL': 150.00, 'GOOG': 1000.00}
        trading_env.historical_peaks = 3000.00

        # Call the calculate_drawdown function
        drawdown = trading_env.calculate_drawdown()

        # Verify the calculated drawdown
        current_value = trading_env.calculate_portfolio_value()
        expected_drawdown = (trading_env.historical_peaks - current_value) / trading_env.historical_peaks
        assert drawdown == expected_drawdown

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

    def test_calculate_indicators(self):
        # Instantiate the TradingEnv object
        trading_env = TradingEnv()

        # Setup
        trading_env.chosen_indicators = {'sma': {'period': 30}, 'rsi': {'period': 14}}
        trading_env.market_data = pd.DataFrame({
            'symbol1': np.random.normal(100, 1, 50), 
            'symbol2': np.random.normal(200, 2, 50)
        })

        # Store a copy of original market data for comparison
        original_market_data = trading_env.market_data.copy(deep=True)

        # Calculate indicators
        trading_env.calculate_indicators()

        # Check that the market data has been updated
        assert not trading_env.market_data.equals(original_market_data)
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