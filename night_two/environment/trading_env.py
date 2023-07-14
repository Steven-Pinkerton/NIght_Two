import numpy as np
import pandas as pd
from night_two.data_handling.data_preprocessing import preprocess_data
from night_two.data_handling.indicators import calculate_ad, calculate_adx, calculate_aroon, calculate_atr, calculate_atr_bands, calculate_atr_trailing_stops, calculate_bbands, calculate_cci, calculate_chaikin_money_flow, calculate_chaikin_oscillator, calculate_chaikin_volatility, calculate_cmo, calculate_coppock, calculate_donchian_channels, calculate_dpo, calculate_elder_ray_index, calculate_ema, calculate_force_index, calculate_hull_moving_average, calculate_ichimoku, calculate_keltner_channels, calculate_kst, calculate_linear_regression, calculate_macd, calculate_psar, calculate_pvt, calculate_rainbow_moving_averages, calculate_rsi, calculate_sma, calculate_standard_deviation_channels, calculate_tmf, calculate_trange, calculate_twiggs_momentum_oscillator, calculate_twiggs_trend_index, calculate_vol_oscillator, calculate_wilder_moving_average, calculate_williams_ad, calculate_wma 

class TradingEnvironment:
    def __init__(self, initial_cash_balance=10000.0, transaction_cost=0.01, data_source='russell_2000_daily.csv'):
        # Define all available indicators and their default settings
        
        self.INDICATORS = {
            'sma': {'func': calculate_sma, 'params': {'period': None}},
            'rsi': {'func': calculate_rsi, 'params': {'period': None}},
            'bbands': {'func': calculate_bbands, 'params': {'period': None}},
            'macd': {'func': calculate_macd, 'params': {'fastperiod': None, 'slowperiod': None, 'signalperiod': None}},
            'psar': {'func': calculate_psar, 'params': {'acceleration': None, 'maximum': None}},
            'trange': {'func': calculate_trange, 'params': {}},
            'wma': {'func': calculate_wma, 'params': {'period': None}},
            'ema': {'func': calculate_ema, 'params': {'period': None}},
            'aroon': {'func': calculate_aroon, 'params': {'period': None}},
            'atr': {'func': calculate_atr, 'params': {'period': None}},
            'ad': {'func': calculate_ad, 'params': {}},
            'adx': {'func': calculate_adx, 'params': {'period': None}},
            'ichimoku': {'func': calculate_ichimoku, 'params': {}},
            'atr_trailing_stops': {'func': calculate_atr_trailing_stops, 'params': {'high': None, 'low': None, 'close': None}},
            'linear_regression': {'func': calculate_linear_regression, 'params': {}},
            'cmo': {'func': calculate_cmo, 'params': {'period': None}},
            'dpo': {'func': calculate_dpo, 'params': {'period': None}},
            'vol_oscillator': {'func': calculate_vol_oscillator, 'params': {'short_period': None, 'long_period': None}},
            'williams_ad': {'func': calculate_williams_ad, 'params': {}},
            'cci': {'func': calculate_cci, 'params': {'timeperiod': None}},
            'pvt': {'func': calculate_pvt, 'params': {}},
            'tmf': {'func': calculate_tmf, 'params': {'timeperiod': None}},
            'donchian_channels': {'func': calculate_donchian_channels, 'params': {'n': None}},
            'keltner_channels': {'func': calculate_keltner_channels, 'params': {'n': None}},
            'atr_bands': {'func': calculate_atr_bands, 'params': {'n': None}},
            'elder_ray_index': {'func': calculate_elder_ray_index, 'params': {'n': None}},
            'hull_moving_average': {'func': calculate_hull_moving_average, 'params': {'n': None}},
            'rainbow_moving_averages': {'func': calculate_rainbow_moving_averages, 'params': {'periods': None}},
            'chaikin_money_flow': {'func': calculate_chaikin_money_flow, 'params': {'n': None}},
            'chaikin_oscillator': {'func': calculate_chaikin_oscillator, 'params': {}},
            'chaikin_volatility': {'func': calculate_chaikin_volatility, 'params': {'n': None}},
            'standard_deviation_channels': {'func': calculate_standard_deviation_channels, 'params': {'n': None}},
            'wilder_moving_average': {'func': calculate_wilder_moving_average, 'params': {'n': None}},
            'twiggs_momentum_oscillator': {'func': calculate_twiggs_momentum_oscillator, 'params': {'n': None}},
            'twiggs_trend_index': {'func': calculate_twiggs_trend_index, 'params': {'n': None}},
            'atr_trailing_stops': {'func': calculate_atr_trailing_stops, 'params': {'high': None, 'low': None, 'close': None, 'atr_period': None, 'multiplier': None}},
            'linear_regression': {'func': calculate_linear_regression, 'params': {'window': None}},
            'coppock': {'func': calculate_coppock, 'params': {'short_roc_period': None, 'long_roc_period': None, 'wma_period': None}},
            'kst': {'func': calculate_kst, 'params': {'rc1': None, 'rc2': None, 'rc3': None, 'rc4': None, 'sma1': None, 'sma2': None, 'sma3': None, 'sma4': None}},
            'force_index': {'func': calculate_force_index, 'params': {'period': None}}
        }

        self.all_indicators = {
            'sma': {'period': 30},
            'rsi': {'period': 14},
            'bbands': {'period': 20},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'psar': {'acceleration': 0.02, 'maximum': 0.2},
            'trange': {},
            'wma': {'period': 30},
            'ema': {'period': 15},
            'aroon': {'period': 14},
            'atr': {'period': 14},
            'ad': {},
            'adx': {'period': 14},
            'ichimoku': {'base_line_periods': 9, 'conversion_line_periods': 26, 'leading_span_b_periods': 52, 'leading_span_a_periods': 26},
            'atr_trailing_stops': {'high': 10, 'low': 10, 'close': 10}, # This might need adjustment
            'linear_regression': {},
            'cmo': {'period': 14},
            'dpo': {'period': 20},
            'vol_oscillator': {'short_period': 5, 'long_period': 20},
            'williams_ad': {},
            'cci': {'timeperiod': 20},
            'pvt': {},
            'tmf': {'timeperiod': 21},
            'donchian_channels': {'n': 20},
            'keltner_channels': {'n': 20},
            'atr_bands': {'n': 14},
            'elder_ray_index': {'n': 14},
            'hull_moving_average': {'n': 9},
            'rainbow_moving_averages': {'periods': 14},
            'chaikin_money_flow': {'n': 20},
            'chaikin_oscillator': {'short_ema_length': 3, 'long_ema_length': 10},
            'chaikin_volatility': {'n': 10},
            'standard_deviation_channels': {'n': 20},
            'wilder_moving_average': {'n': 14},
            'twiggs_momentum_oscillator': {'n': 21},
            'twiggs_trend_index': {'n': 21},
            'atr_trailing_stops': {'high': 10, 'low': 10, 'close': 10, 'atr_period': 14, 'multiplier': 2},
            'linear_regression': {'window': 14},
            'coppock': {'short_roc_period': 11, 'long_roc_period': 14, 'wma_period': 10},
            'kst': {'rc1': 10, 'rc2': 15, 'rc3': 20, 'rc4': 30, 'sma1': 10, 'sma2': 10, 'sma3': 10, 'sma4': 15},
            'force_index': {'period': 13},
        }
        
        # Assign all indicators to chosen_indicators
        self.chosen_indicators = self.all_indicators
        
        
        # Initialize params_values as a copy of all_indicators
        self.params_values = self.all_indicators.copy()
        
        # Initialize the indicators
        self.indicator_values = {name: None for name in self.all_indicators.keys()}

        # Initialize the data
        self.data = self.load_market_data(data_source)
        self.original_market_data = self.data.copy()  # store the original data
        self.current_step = 0
        self.initial_cash_balance = initial_cash_balance

        # Initialize account balances and transactions costs
        self.cash_balance = initial_cash_balance
        self.transaction_cost = transaction_cost
        self.portfolio = defaultdict(float)
            

        self.risk_adjusted_return_history = []
        self.portfolio_value_history = []

        self.max_action = self.calculate_max_action()

    def calculate_max_action(self):
         # Define the possible percentages
        percentages = list(range(10, 110, 10))  # 10%, 20%, ..., 100%

         # Define the action space
        actions = self.define_action_space()

        # Calculate max_action
        # For each stock, there are 4 possible actions (Buy, Sell, Hold, Change Settings) and 10 possible percentages
        max_action = len(self.market_data.columns) * len(actions) * len(percentages)

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

        # Go through each indicator
        for indicator_name, indicator_info in self.INDICATORS.items():
            # Get the parameters defined in INDICATORS
            indicator_params = indicator_info['params']
            
            # Go through each defined parameter
            for param_name in indicator_params.keys():
                if param_name in data_series.keys():
                    # Assign the corresponding data series based on the parameter name
                    indicator_info['params'][param_name] = data_series[param_name]
                else:
                    # Get the default parameter value from all_indicators
                    default_param_value = self.all_indicators[indicator_name].get(param_name)
                    if default_param_value is not None:
                        indicator_info['params'][param_name] = default_param_value

        return market_data

    def initialize_state(self):
        # Initialize the portfolio and cash balance
        self.portfolio = {}  # Dict with structure: {"Stock Symbol": Number of shares held}
        self.cash_balance = self.initial_cash_balance

        # Initialize the market state
        self.market_state = self.market_data.iloc[0]

        # Initialize performance metrics
        self.performance_metrics = self.calculate_initial_metrics()

        # Initialize chosen indicators with a copy of all indicators
        self.chosen_indicators = self.all_indicators.copy()

        # Initialize previous action
        self.previous_action = None
        
        # Initialize prices and counters
        self.buy_prices = {}  # Structure: {'symbol': [price1, price2, ...]}
        self.sell_prices = {}  # Structure: {'symbol': [price1, price2, ...]}
        self.winning_trades = 0
        self.total_trades = 0

        # Concatenate portfolio state, cash balance and market state into full state
        self.state = self.concatenate_state()

        return self.state

    def concatenate_state(self):
        # Convert portfolio and cash_balance into a compatible format with market_state
        portfolio_vector = self.portfolio_to_vector()
        cash_balance_vector = np.array([self.cash_balance])

        # Convert performance metrics, chosen indicators, and previous action to a compatible format
        performance_vector = self.metrics_to_vector(self.performance_metrics)
        indicator_vector = self.indicator_settings_to_vector(self.chosen_indicators)
        action_vector = self.action_to_vector(self.previous_action)

        # Concatenate all components of the state
        full_state = np.concatenate([portfolio_vector, cash_balance_vector, self.market_state.values, performance_vector, indicator_vector, action_vector])

        return full_state
    
    def portfolio_to_vector(self):
        # Convert portfolio dictionary to a vector (array), compatible with the rest of the state
        # Assume that the stocks are ordered in the same order as in the market_data
        portfolio_vector = []

        for stock_symbol in self.market_data.columns:
            if stock_symbol in self.portfolio:
                portfolio_vector.append(self.portfolio[stock_symbol])
            else:
                portfolio_vector.append(0)

        return np.array(portfolio_vector)
    
    def metrics_to_vector(self, metrics):
        # Convert metrics dictionary to a vector (array), compatible with the rest of the state
        # This function simply extracts the values and forms a numpy array
        metrics_vector = np.array(list(metrics.values()))
        return metrics_vector
    
    def action_to_vector(self, action):
        # Assuming the action is represented as an integer index
        if action is None:
            return 0  # Could be a special value representing "no action"
        else:
            return action

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
        if 'change_indicator_settings' in action:
            # Update technical indicator settings and recalculate market data with new settings
            self.indicator_settings = self.update_indicator_settings(action['change_indicator_settings'])
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
        self.portfolio = {}
        self.cash_balance = self.initial_cash_balance
        self.buy_prices = {}
        self.sell_prices = {}
        self.winning_trades = 0
        self.total_trades = 0
        return self.state

    def calculate_reward(self):
        # Compute the return
        ret = (self.current_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value

        # Compute a measure of risk
        risk = np.std(self.portfolio_value_history)

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

        symbol, action_type, amount = action['symbol'], action['type'], action['amount']
        current_price = self.market_state[symbol]

        if action_type == 'buy':
            cost = current_price * amount

            # If sufficient balance is available, update portfolio and balance
            if self.cash_balance >= cost:
                if symbol not in self.portfolio:
                    self.portfolio[symbol] = 0
                self.portfolio[symbol] += amount
                self.cash_balance -= cost

                # Store buy price
                if symbol not in self.buy_prices:
                    self.buy_prices[symbol] = []
                self.buy_prices[symbol].append(current_price)

        elif action_type == 'sell':
            # If sufficient stocks are available in the portfolio, update portfolio and balance
            if symbol in self.portfolio and self.portfolio[symbol] >= amount:
                self.portfolio[symbol] -= amount
                if self.portfolio[symbol] == 0:
                    del self.portfolio[symbol]
                self.cash_balance += current_price * amount

                # Store sell price and count winning trades
                if symbol not in self.sell_prices:
                    self.sell_prices[symbol] = []
                self.sell_prices[symbol].append(current_price)
                if self.sell_prices[symbol][-1] > self.buy_prices[symbol][0]:  # FIFO strategy
                    self.winning_trades += 1
                del self.buy_prices[symbol][0]  # Remove the corresponding buy price

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

    def is_valid_action(self, action):
        # Check that action is a dictionary containing the necessary keys
        if not isinstance(action, dict):
            return False
        if not {'symbol', 'type', 'amount'}.issubset(action.keys()):
            return False
        
        # Check that action type is in the action space
        if action['type'] not in self.action_space:
            return False

        # Check that the action symbol is in the current market state (i.e., it's a tradable asset)
        if action['symbol'] not in self.market_state.keys():
            return False

        # Check that the action amount is a non-negative number
        if not isinstance(action['amount'], (int, float)) or action['amount'] < 0:
            return False

        return True

    def recalculate_market_data(self):
        # Reset the market data to the original data
        self.market_data = self.original_market_data.copy()

        # Recalculate each indicator in the chosen indicators with the updated settings
        for indicator_name, settings in self.indicator_settings.items():
            # Get the function and parameter names for this indicator
            func = INDICATORS.get(indicator_name)['func']
            param_names = INDICATORS.get(indicator_name)['params']

            # Create a dictionary of parameter values from the updated indicator settings
            func_params = {name: settings[name] for name in param_names}

            # Calculate the indicator and update the market data
            self.market_data = func(self.market_data, **func_params)

        return self.market_data
    
    def update_metrics(self):
        # Update metrics
        current_portfolio_value = self.calculate_portfolio_value()
        running_average_value = (self.performance_metrics['Running Average Value'] * (self.current_step - 1) + current_portfolio_value) / self.current_step
        self.performance_metrics = {
            'Portfolio Value': current_portfolio_value,
            'Running Average Value': running_average_value,
            'Drawdown': self.calculate_drawdown(current_portfolio_value),
            'Winning Trades': self.calculate_winning_trades(),  # This needs more implementation details
            'Total Trades': self.calculate_total_trades()  # This needs more implementation details
        }

    def calculate_portfolio_value(self):
        return sum(self.market_state[symbol] * amount for symbol, amount in self.portfolio.items())

    def calculate_drawdown(self):
        if not hasattr(self, 'historical_peaks'):
            self.historical_peaks = self.calculate_portfolio_value()
        current_value = self.calculate_portfolio_value()
        self.historical_peaks = max(self.historical_peaks, current_value)
        drawdown = (self.historical_peaks - current_value) / self.historical_peaks
        return drawdown

    def calculate_winning_trades(self):
        return self.winning_trades

    def calculate_total_trades(self):
        return self.total_trades

    def update_indicator_settings(self, indicator_name, settings):
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

    def calculate_indicators(self):
        """Calculate the value for all indicators."""

        # Ensure market data is loaded
        if self.indicator_data is None:
            raise Exception("Market data must be loaded before calculating indicators")

        # Iterate over all indicators
        for indicator_name, indicator_info in self.INDICATORS.items():
            # Get the function to calculate the indicator
            indicator_func = indicator_info['func']

            # Iterate over all parameters for this indicator
            params = {}
            for param_name in indicator_info['params'].keys():
                # Calculate the value for this parameter
                param_value = self.calculate_parameter_value(indicator_name, param_name)

                # Store the calculated parameter value
                params[param_name] = param_value

            # Calculate the indicator value with the determined parameters
            indicator_value = indicator_func(self.indicator_data, **params)

            # Store the calculated indicator value
            self.indicator_values[indicator_name] = indicator_value

                    
    def calculate_parameter_value(self, indicator_name, param_name):
        """Calculate the value for a specific parameter of a specific indicator."""

        # Check if this parameter has a value set in params_values
        if self.params_values and indicator_name in self.params_values and param_name in self.params_values[indicator_name]:
            return self.params_values[indicator_name][param_name]

        # If not, use the default value from all_indicators
        if indicator_name in self.all_indicators and param_name in self.all_indicators[indicator_name]:
            return self.all_indicators[indicator_name][param_name]

        # If there is no default value, raise an exception
        raise Exception(f"Cannot calculate value for parameter '{param_name}' of indicator '{indicator_name}'")
                
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
    