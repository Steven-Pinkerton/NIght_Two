import numpy as np
import pandas as pd
from collections import defaultdict
from night_two.data_handling.data_preprocessing import preprocess_data
from night_two.data_handling.indicators import calculate_ad, calculate_adx, calculate_aroon, calculate_atr, calculate_atr_bands, calculate_bbands, calculate_cci, calculate_chaikin_money_flow, calculate_chaikin_oscillator, calculate_chaikin_volatility, calculate_cmo, calculate_coppock, calculate_donchian_channels, calculate_dpo, calculate_elder_ray_index, calculate_ema, calculate_force_index, calculate_hull_moving_average, calculate_ichimoku, calculate_keltner_channels, calculate_kst, calculate_linear_regression, calculate_macd, calculate_psar, calculate_pvt, calculate_rainbow_moving_averages, calculate_rsi, calculate_sma, calculate_standard_deviation_channels, calculate_tmf, calculate_trange, calculate_twiggs_momentum_oscillator, calculate_twiggs_trend_index, calculate_vol_oscillator, calculate_wilder_moving_average, calculate_williams_ad, calculate_wma 

class TradingEnvironment:
    def __init__(self, initial_cash_balance=10000.0, transaction_cost=0.01, data_source='russell_2000_daily.csv'):
        # Define all available indicators and their default settings
        
        self.INDICATORS = {
            'sma': {'func': calculate_sma, 'params': {'period': range(10, 51)}},
            'rsi': {'func': calculate_rsi, 'params': {'period': range(5, 31)}},
            'bbands': {'func': calculate_bbands, 'params': {'period': range(10, 51)}},
            'macd': {'func': calculate_macd, 'params': {'fastperiod': range(10, 26), 'slowperiod': range(27, 51), 'signalperiod': range(5, 10)}},
            'psar': {'func': calculate_psar, 'params': {'acceleration': [x * 0.01 for x in range(2, 21)], 'maximum': [x * 0.01 for x in range(2, 21)]}},
            'wma': {'func': calculate_wma, 'params': {'period': range(10, 51)}},
            'ema': {'func': calculate_ema, 'params': {'period': range(10, 51)}},
            'aroon': {'func': calculate_aroon, 'params': {'period': range(5, 31)}},
            'atr': {'func': calculate_atr, 'params': {'period': range(5, 31)}},
            'adx': {'func': calculate_adx, 'params': {'period': range(5, 31)}},
            'cmo': {'func': calculate_cmo, 'params': {'period': range(5, 31)}},
            'dpo': {'func': calculate_dpo, 'params': {'period': range(5, 31)}},
            'vol_oscillator': {'func': calculate_vol_oscillator, 'params': {'short_period': range(3, 16), 'long_period': range(17, 31)}},
            'cci': {'func': calculate_cci, 'params': {'timeperiod': range(10, 51)}},
            'tmf': {'func': calculate_tmf, 'params': {'timeperiod': range(10, 31)}},
            'donchian_channels': {'func': calculate_donchian_channels, 'params': {'n': range(10, 51)}},
            'keltner_channels': {'func': calculate_keltner_channels, 'params': {'n': range(10, 51)}},
            'atr_bands': {'func': calculate_atr_bands, 'params': {'n': range(10, 51)}},
            'elder_ray_index': {'func': calculate_elder_ray_index, 'params': {'n': range(10, 51)}},
            'hull_moving_average': {'func': calculate_hull_moving_average, 'params': {'n': range(5, 26)}},
            'rainbow_moving_averages': {'func': calculate_rainbow_moving_averages, 'params': {'periods': range(5, 31)}},
            'chaikin_money_flow': {'func': calculate_chaikin_money_flow, 'params': {'n': range(10, 51)}},
            'chaikin_volatility': {'func': calculate_chaikin_volatility, 'params': {'n': range(10, 51)}},
            'standard_deviation_channels': {'func': calculate_standard_deviation_channels, 'params': {'n': range(10, 51)}},
            'wilder_moving_average': {'func': calculate_wilder_moving_average, 'params': {'n': range(5, 31)}},
            'twiggs_momentum_oscillator': {'func': calculate_twiggs_momentum_oscillator, 'params': {'n': range(10, 51)}},
            'twiggs_trend_index': {'func': calculate_twiggs_trend_index, 'params': {'n': range(10, 51)}},
            'linear_regression': {'func': calculate_linear_regression, 'params': {'window': range(5, 31)}},
            'coppock': {'func': calculate_coppock, 'params': {'short_roc_period': range(10, 21), 'long_roc_period': range(22, 31), 'wma_period': range(5, 16)}},
            'kst': {'func': calculate_kst, 'params': {'rc1': range(5, 16), 'rc2': range(10, 21), 'rc3': range(15, 26), 'rc4': range(20, 31), 'sma1': range(5, 16), 'sma2': range(10, 21), 'sma3': range(15, 26), 'sma4': range(20, 31)}},
            'force_index': {'func': calculate_force_index, 'params': {'period': range(5, 31)}},
            # other indicators...
        }
        
        # Assign all indicators to chosen_indicators
        self.chosen_indicators = {}
        self.all_indicators = self.initialize_all_indicators()
        
        

        # Initialize params_values as a copy of all_indicators
        self.params_values = self.all_indicators.copy()
        
        # Initialize the indicators
        self.indicator_values = {name: None for name in self.all_indicators.keys()}

        # Initialize the data
        self.data = self.load_market_data(data_source)
        self.indicator_data = self.data
        self.original_market_data = self.data.copy()  # store the original data
        self.current_step = 0
        self.initial_cash_balance = initial_cash_balance

        # Initialize account balances and transactions costs
        self.cash_balance = initial_cash_balance
        self.transaction_cost = transaction_cost
        self.shares = []
            
        self.risk_adjusted_return_history = []
        self.portfolio_value_history = []

        self.max_action = self.calculate_max_action()

    def calculate_max_action(self):
        # Define the possible percentages
        percentages = [i for i in range(0, 101, 10)]

        # Define the action space
        actions = ['buy', 'sell', 'hold']

        # Calculate max_action
        # For each stock, there are 3 possible actions (Buy, Sell, Hold) and 10 possible percentages
        max_action = len(actions) * len(percentages)

        # Additional actions for changing settings of each indicator
        indicator_actions = 0
        for indicator in self.INDICATORS.values():
            params = indicator['params']
            for param_values in params.values():
                # Add the number of possible values for each parameter to the indicator_actions
                indicator_actions += len(param_values)

        # Adding indicator actions to the max action
        max_action += indicator_actions

        return max_action

    def load_market_data(self, data_source):
        # Load the market data from the data source
        market_data = pd.read_csv(data_source)

        # Extract the required Series from the DataFrame
        data_series = {
            'high': market_data['High'],
            'low': market_data['Low'],
            'close': market_data['Close'],
            'volume': market_data['Volume']
        }

        return market_data

    def initialize_state(self):
        # Initialize the portfolio with 0 shares
        self.num_shares = 0

        # Initialize the cash balance
        self.cash_balance = self.initial_cash_balance

        # Initialize the market state
        self.market_state = self.market_data.iloc[0]

        # Initialize performance metrics
        self.performance_metrics = self.calculate_initial_metrics()

        # Initialize chosen indicators as empty dictionary
        self.chosen_indicators = {}

        # Initialize previous action
        self.previous_action = None

        # Initialize prices and counters
        self.buy_price = 0  # Price at which the asset was last bought
        self.sell_price = 0  # Price at which the asset was last sold
        self.winning_trades = 0
        self.total_trades = 0

        # Concatenate portfolio state, cash balance and market state into full state
        self.state = self.concatenate_state()

        return self.state

    def concatenate_state(self):
        # Convert num_shares and cash_balance into a compatible format with market_state
        num_shares_vector = np.array([self.num_shares])
        cash_balance_vector = np.array([self.cash_balance])

        # Convert performance metrics, chosen indicators, and previous action to a compatible format
        performance_vector = self.metrics_to_vector(self.performance_metrics)
        indicator_vector = self.indicator_settings_to_vector(self.chosen_indicators)
        action_vector = self.action_to_vector(self.previous_action)

        # Concatenate all components of the state
        full_state = np.concatenate([num_shares_vector, cash_balance_vector, self.market_state.values, performance_vector, indicator_vector, action_vector])

        return full_state

    def metrics_to_vector(self, metrics):
        # Convert metrics dictionary to a vector (array), compatible with the rest of the state
        # This function simply extracts the values and forms a numpy array
        metrics_vector = np.array(list(metrics.values()))
        return metrics_vector
    
    def action_to_vector(self, action):
        # Assuming the action is represented as a string
        if action is None:
            return np.array([0])  # Could be a special value representing "no action"
        else:
            # Convert the action into an index in the action space
            action_index = self.action_space.index(action)
            return np.array([action_index])
        
    def define_action_space(self):
        # Define a discrete action space
        self.action_space = ['Buy', 'Sell', 'Hold', 'Change Settings']
        return self.action_space

    def step(self, action):
        # Copy the current portfolio value
        self.previous_portfolio_value = self.current_portfolio_value
        
        # Update portfolio and cash balance based on the action
        self.update_portfolio_and_balance(action)

        # Update the current portfolio value
        self.current_portfolio_value = self.calculate_portfolio_value()

        # Update performance metrics
        self.performance_metrics = self.update_metrics()

       # Check if the action involved changing indicator settings
        if action == 'change_indicator_settings':
            # Update technical indicator settings and recalculate market data with new settings
            self.chosen_indicators = self.update_indicator_settings()
            self.market_data = self.recalculate_market_data()
        
        # Update previous action
        self.previous_action = action

        # Get the new market state for the next time step
        self.current_step += 1
        self.market_state = self.market_data.iloc[self.current_step]

        # Update full state
        self.state = self.concatenate_state()

        # Calculate the reward
        reward = self.calculate_reward()

        # Check if the episode has ended
        done = self.current_step >= len(self.market_data) or self.cash_balance <= 0

        return self.state, reward, done

    def reset(self):
        # Reset the environment to the initial state
        self.state = self.initialize_state()
        self.current_step = 0
        self.portfolio = 0
        self.cash_balance = self.initial_cash_balance
        self.buy_price = 0
        self.sell_price = 0
        self.winning_trades = 0
        self.total_trades = 0
        return self.state

    def calculate_reward(self):
            # Compute the return
            ret = (self.current_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value

            # Compute a measure of risk
            risk = np.std(self.portfolio_value_history)

            # Add a small constant to the risk to prevent division by zero
            EPSILON = 1e-8
            risk += EPSILON

            # Compute the risk-adjusted return
            risk_adjusted_return = ret / risk

            # Compute a penalty for trading
            trade_penalty = self.transaction_cost * (self.current_portfolio_value != self.previous_portfolio_value)

            # Compute an improvement bonus
            improvement_bonus = 0.0
            lookback_period = 10  # Define the period over which improvement is measured
            if self.current_step > lookback_period:
                old_risk_adjusted_return = self.risk_adjusted_return_history[-lookback_period]
                old_portfolio_value = self.portfolio_value_history[-lookback_period]
                if risk_adjusted_return > old_risk_adjusted_return and self.current_portfolio_value > old_portfolio_value:
                    improvement_bonus = 0.1  # This value could be adjusted as per requirements

            # The reward is the risk-adjusted return minus the trade penalty plus the improvement bonus
            reward = risk_adjusted_return - trade_penalty + improvement_bonus

            # Store current risk-adjusted return and portfolio value for future comparisons
            self.risk_adjusted_return_history.append(risk_adjusted_return)
            self.portfolio_value_history.append(self.current_portfolio_value)

            return reward
    
    def update_portfolio_and_balance(self, action):
            # Check the validity of the action
            if not self.is_valid_action(action):
                return

            action_type, amount = action['type'], action['amount']
            current_price = self.market_state[self.symbol]

            if action_type == 'buy':
                cost = current_price * amount

                # If sufficient balance is available, update portfolio and balance
                if self.cash_balance >= cost:
                    self.num_shares += amount
                    self.cash_balance -= cost

                    # Store buy price
                    self.buy_price = current_price

            elif action_type == 'sell':
                # If sufficient stocks are available in the portfolio, update portfolio and balance
                if self.num_shares >= amount:
                    self.num_shares -= amount
                    self.cash_balance += current_price * amount

                    # Store sell price and count winning trades
                    self.sell_price = current_price
                    if self.sell_price > self.buy_price:  # FIFO strategy
                        self.winning_trades += 1

            # Increment total trades counter
            self.total_trades += 1

            self.update_market_state()  # Update the market state after taking the action

    def calculate_initial_metrics(self):
        # Calculate initial metrics
        self.performance_metrics = {
            'Portfolio Value': self.calculate_portfolio_value(),
            'Running Average Value': self.calculate_portfolio_value(),  # Add this new metric
            'Drawdown': 0,  # No drawdown at the start
            'Winning Trades': 0,  # No trades at the start
            'Total Trades': 0  # No trades at the start
        }

    def update_market_state(self):
        if self.current_step < len(self.market_data) - 1:
            # If we haven't stepped past the latest available data, advance to the next time step
            self.current_step += 1
            # Else, we maintain the current_step to be the last index of the market_data

        # Update market_state to reflect the data at the current step
        self.market_state = self.market_data.iloc[self.current_step]

    def recalculate_market_data(self):
        # Reset the market data to the original data
        self.market_data = self.original_market_data.copy()

        # Recalculate each indicator in the chosen indicators with the updated settings
        for indicator_name, settings in self.chosen_indicators.items():
            # Get the function for this indicator
            indicator_func = self.INDICATORS[indicator_name]['func']

            # Calculate the indicator and update the market data
            self.market_data[indicator_name] = indicator_func(self.market_data, **settings)

        return self.market_data
    
    def update_metrics(self):
        # Update metrics
        current_portfolio_value = self.calculate_portfolio_value()
        running_average_value = (self.performance_metrics['Running Average Value'] * (self.current_step - 1) + current_portfolio_value) / self.current_step
        self.performance_metrics = {
            'Portfolio Value': current_portfolio_value,
            'Running Average Value': running_average_value,
            'Drawdown': self.calculate_drawdown(current_portfolio_value),
            'Winning Trades': self.winning_trades, 
            'Total Trades': self.total_trades 
        }

    def calculate_portfolio_value(self):
        # Assuming 'close' is the closing price of the asset
        return self.market_state['close'] * self.num_shares

    def calculate_drawdown(self, current_value):
        if not hasattr(self, 'historical_peaks'):
            self.historical_peaks = current_value
        self.historical_peaks = max(self.historical_peaks, current_value)
        drawdown = (self.historical_peaks - current_value) / self.historical_peaks
        return drawdown

    def calculate_winning_trades(self):
        return self.winning_trades

    def calculate_total_trades(self):
        return self.total_trades

    def update_indicator_settings(self, settings_dict):
        for indicator_name, settings in settings_dict.items():
            if indicator_name in self.all_indicators and len(self.all_indicators[indicator_name]) > 0:
                # the indicator is present in the all_indicators dictionary and it has adjustable parameters
                self.chosen_indicators[indicator_name] = settings
            elif indicator_name in INDICATORS:
                # the indicator is present in the INDICATORS dictionary but it has no adjustable parameters
                # so we just select the indicator without updating any settings
                self.chosen_indicators[indicator_name] = {}
            else:
                # the indicator is not recognized
                raise ValueError(f"Unknown indicator: {indicator_name}")
                    
    def indicator_settings_to_vector(self, settings):
        # Convert indicator settings to a vector (array)
        # For simplicity, let's say settings are represented by a single scalar value for each indicator
        settings_vector = np.array(list(settings.values()))
        return settings_vector

    def calculate_indicators(self, params_values=None):
        """Calculate the value for all chosen indicators."""

        # Ensure market data is loaded
        if self.indicator_data is None:
            raise Exception("Market data must be loaded before calculating indicators")

        # If no new parameter values provided, use existing ones
        params_values = params_values if params_values is not None else self.params_values

        # Iterate over all chosen indicators
        for indicator_name, settings in self.chosen_indicators.items():
            # Get the function to calculate the indicator
            indicator_func = self.INDICATORS[indicator_name]['func']

            # Get parameters for this indicator from provided values
            params = params_values.get(indicator_name, {})

            # Calculate the indicator value with the determined parameters
            indicator_value = indicator_func(self.indicator_data, **params)

            # Store the calculated indicator value
            self.indicator_values[indicator_name] = indicator_value
  
               
    def select_indicators(self, chosen_indicators):
        """
        Method for the agent to select which indicators to use. This can be
        called as part of the agent's initialization, or as an action the agent
        can take.
        """
        # Validate the chosen indicators
        for indicator_name in chosen_indicators:
            if indicator_name not in self.all_indicators:
                raise ValueError(f"Unknown indicator: {indicator_name}")
        
        # Use the `update_indicator_settings` method to initialize settings for
        # the chosen indicators
        self.chosen_indicators = self.update_indicator_settings(chosen_indicators)
        
    def run_episode(self):
        self.env.reset()  # reset the environment to its initial state at the start of an episode
        done = False  # flag to track if the episode has ended
        total_reward = 0  # To keep track of total reward in the episode

        while not done:
            action = self.agent.get_action(self.state)  # agent selects an action based on the current state
            next_state, reward, done, _ = self.env.step(action)  # environment transitions to the next state based on the action, and provides a reward

            self.agent.learn(self.state, action, reward, next_state, done)  # agent updates its strategy based on the experience

            self.state = next_state  # update the current state to the next state
            total_reward += reward  # Add reward to total reward

            if self.balance <= 0 or self.step == len(self.data) - self.sequence_length:  # check if the agent has run out of funds or data
                done = True  # if so, the episode is done

        return total_reward  # return the total reward from the episode
    