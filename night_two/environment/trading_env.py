import copy
import numpy as np
from typing import List, Optional, Dict, Union, Any, Tuple, Callable
import random
import pandas as pd
from collections import defaultdict
from night_two.data_handling.data_preprocessing import preprocess_data
from night_two.data_handling.indicators import calculate_ad, calculate_adx, calculate_aroon, calculate_atr, calculate_atr_bands, calculate_bbands, calculate_cci, calculate_chaikin_money_flow, calculate_chaikin_oscillator, calculate_chaikin_volatility, calculate_cmo, calculate_coppock, calculate_donchian_channels, calculate_dpo, calculate_elder_ray_index, calculate_ema, calculate_force_index, calculate_hull_moving_average, calculate_ichimoku, calculate_keltner_channels, calculate_kst, calculate_linear_regression, calculate_macd, calculate_psar, calculate_pvt, calculate_rainbow_moving_averages, calculate_rsi, calculate_sma, calculate_standard_deviation_channels, calculate_tmf, calculate_trange, calculate_twiggs_momentum_oscillator, calculate_twiggs_trend_index, calculate_vol_oscillator, calculate_wilder_moving_average, calculate_williams_ad, calculate_wma 

class TradingEnvironment:
    def __init__(self, initial_cash_balance: float = 10000.0, transaction_cost: float = 0.01, data_source: str = 'russell_2000_daily.csv') -> None:
        """
        Initialize a new instance of the trading environment.

        :param initial_cash_balance: The initial amount of cash available.
        :param transaction_cost: The cost per transaction.
        :param data_source: The source of the market data.
        """

        # Initialize the cash balance and transaction costs.
        self.cash_balance = initial_cash_balance
        self.initial_cash_balance = initial_cash_balance
        self.transaction_cost = transaction_cost


        # Initialize the data, loading it from the specified data source.
        self.data = self.load_market_data(data_source)

        # Store a copy of the original market data.
        self.original_market_data = self.data.copy()

        # Initialize other attributes.
        self._initialize_attributes()

        # Initialize the action and observation spaces.
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space(len(self.data.columns) + 2)

    def _initialize_attributes(self) -> None:
        """
        Initialize various attributes used by the environment.
        """
        # Define all available indicators and their default settings.
        self._initialize_indicators()
        
        self.last_action = None
        
        # Initialize the last trade value.
        self.last_trade_value = 0.0

        # Initialize the params_values as a copy of INDICATORS.
        self.params_values = copy.deepcopy(self.INDICATORS)
        
        # Initialize actual_params_values with the same keys as INDICATORS
        self.actual_params_values = {indicator: {} for indicator in self.INDICATORS.keys()}

        # Initialize portfolio values.
        self.current_portfolio_value = 0
        self.previous_portfolio_value = 0
        self.historical_peaks = 0

        # Initialize the number of shares and the total trades.
        self.num_shares = 0
        self.total_trades = 0

        # Initialize the risk-adjusted return and portfolio value history.
        self.risk_adjusted_return_history = []
        self.portfolio_value_history = []

        # Initialize the state.
        self.initialize_state()

    def _initialize_indicators(self) -> None:
        """
        Initialize the indicators and their settings.
        """

        # Define all available indicators and their default settings.
        self.INDICATORS = {
            'sma': {'func': calculate_sma, 'params': {'period': range(1, 500)}},
            'rsi': {'func': calculate_rsi, 'params': {'period': range(1, 500)}},
            'bbands': {'func': calculate_bbands, 'params': {'period': range(1, 500)}},
            'macd': {'func': calculate_macd, 'params': {'fastperiod': range(1, 500), 'slowperiod': range(1, 500), 'signalperiod': range(1, 500)}},
            'psar': {'func': calculate_psar, 'params': {'acceleration': range(0, 10), 'maximum': range(0, 10)}},
            'trange': {'func': calculate_trange, 'params': {}},
            'wma': {'func': calculate_wma, 'params': {'period': range(1, 500)}},
            'ema': {'func': calculate_ema, 'params': {'period': range(1, 500)}},
            'aroon': {'func': calculate_aroon, 'params': {'period': range(1, 500)}},
            'atr': {'func': calculate_atr, 'params': {'period': range(1, 500)}},
            'ad': {'func': calculate_ad, 'params': {}},
            'adx': {'func': calculate_adx, 'params': {'period': range(1, 500)}},
            'ichimoku': {'func': calculate_ichimoku, 'params': {}},
            'linear_regression': {'func': calculate_linear_regression, 'params': {}},
            'cmo': {'func': calculate_cmo, 'params': {'period': range(1, 500)}},
            'dpo': {'func': calculate_dpo, 'params': {'period': range(1, 500)}},
            'vol_oscillator': {'func': calculate_vol_oscillator, 'params': {'short_period': range(1, 500), 'long_period': range(1, 500)}},
            'williams_ad': {'func': calculate_williams_ad, 'params': {}},
            'cci': {'func': calculate_cci, 'params': {'timeperiod': range(1, 500)}},
            'pvt': {'func': calculate_pvt, 'params': {}},
            'tmf': {'func': calculate_tmf, 'params': {'timeperiod': range(1, 500)}},
            'donchian_channels': {'func': calculate_donchian_channels, 'params': {'n': range(1, 500)}},
            'keltner_channels': {'func': calculate_keltner_channels, 'params': {'n': range(1, 500)}},
            'atr_bands': {'func': calculate_atr_bands, 'params': {'n': range(1, 500)}},
            'elder_ray_index': {'func': calculate_elder_ray_index, 'params': {'n': range(1, 500)}},
            'hull_moving_average': {'func': calculate_hull_moving_average, 'params': {'n': range(1, 500)}},
            'rainbow_moving_averages': {'func': calculate_rainbow_moving_averages, 'params': {'periods': range(1, 500)}},
            'chaikin_money_flow': {'func': calculate_chaikin_money_flow, 'params': {'n': range(1, 500)}},
            'chaikin_oscillator': {'func': calculate_chaikin_oscillator, 'params': {}},
            'chaikin_volatility': {'func': calculate_chaikin_volatility, 'params': {'n': range(1, 500)}},
            'standard_deviation_channels': {'func': calculate_standard_deviation_channels, 'params': {'n': range(1, 500)}},
            'wilder_moving_average': {'func': calculate_wilder_moving_average, 'params': {'n': range(1, 500)}},
            'twiggs_momentum_oscillator': {'func': calculate_twiggs_momentum_oscillator, 'params': {'n': range(1, 500)}},
            'twiggs_trend_index': {'func': calculate_twiggs_trend_index, 'params': {'n': range(1, 500)}},
            'linear_regression': {'func': calculate_linear_regression, 'params': {'window': range(1, 500)}},
            'coppock': {'func': calculate_coppock, 'params': {'short_roc_period': range(1, 500), 'long_roc_period': range(1, 500), 'wma_period': range(1, 500)}},
            'kst': {'func': calculate_kst, 'params': {'rc1': range(1, 500), 'rc2': range(1, 500), 'rc3': range(1, 500), 'rc4': range(1, 500), 'sma1': range(1, 500), 'sma2': range(1, 500), 'sma3': range(1, 500), 'sma4': range(1, 500)}},
            'force_index': {'func': calculate_force_index, 'params': {'period': range(1, 500)}}
        }

        # Initialize indicator_settings as an empty dictionary.
        self.indicator_settings = {}

        # Assign all indicators to chosen_indicators.
        self.chosen_indicators = {}

        # Initialize params_values as a copy of INDICATORS.
        self.params_values = copy.deepcopy(self.INDICATORS)

        # Initialize the indicators.
        self.indicator_values = {name: None for name in self.INDICATORS.keys()}

    def _define_observation_space(self, size: int) -> List[Optional[float]]:
        # Define the observation space for the trading environment. 
        # An observation space represents the state of the environment that the agent observes.

        # Parameters:
        # size (int): The size of the observation space.

        # Returns:
        # List[Optional[float]]: The observation space, which is a list of size `size` 
        #                    with all elements as `None`, signifying that any real number 
        #                    is an acceptable value in the state vector.
        

        # If your state is a vector of continuous variables, you can represent this
        # as a list of 'None' with the appropriate length. This signifies that any
        # real number is an acceptable value in the state vector.
        return [None] * size

    def _define_action_space(self) -> List[str]: 
        # Define the action space for the trading environment. 
        # An action space consists of all possible actions that an agent can take in the environment.

        # Returns:
        # List[str]: The action space, which is a list of the possible actions that 
        #        the agent can take. In this case, it includes 'buy', 'sell', 'hold', 
        #        'change_indicator_settings', and 'select_indicators'.

        # Define the action space as a list of all possible actions.
        return ['buy', 'sell', 'hold', 'change_indicator_settings', 'select_indicators']

    def calculate_max_action(self) -> int:
    # Calculate the maximum number of actions that the trading agent can take. 
    # This includes both trade actions and actions to change indicator settings.

    # Returns:
    # int: The total number of possible actions. 

    # Define the possible percentages for shares to buy or sell
        percentages = [i for i in range(1, 101)]  # range from 1 to 100

        # Define the action space
        actions = ['buy', 'sell', 'hold']

        # Calculate max_action
        # For each stock, there are 3 possible actions (Buy, Sell, Hold) and 100 possible percentages
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

    def load_market_data(self, data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        
        #Load market data from a CSV file or directly from a DataFrame.
        # 
        # Parameters:
        # data_source (Union[str, pd.DataFrame]): Either a path to the CSV file containing the market data, or
        #                                    a pandas DataFrame object containing the market data.

        #Returns:
        #pd.DataFrame: DataFrame containing the loaded market data. 

        #Raises:
        #ValueError: If the data_source is not a string (assumed to be a file path) or a pandas DataFrame.

        # Check the type of the data_source
        if isinstance(data_source, pd.DataFrame):
            # If data_source is a DataFrame, use it directly
            market_data = data_source
        elif isinstance(data_source, str):
            # If data_source is a string, attempt to load a CSV file from the path specified
            market_data = pd.read_csv(data_source)
        else:
            # If data_source is neither a DataFrame nor a string, raise an error
            raise ValueError("Invalid type for data_source. Must be a pandas DataFrame or a string path to a CSV file.")
        
        # Return the loaded market data
        return market_data

    def concatenate_state(self) -> np.ndarray:
        """
        Combine various components of the environment state into a single numpy array. 
        
        :return: The full state of the environment, including the number of shares, cash balance, market state,
                performance metrics, chosen indicators, and prices of the asset when it was last bought or sold.
        """

        # Convert single value components to numpy arrays
        num_shares_vector = np.array([self.num_shares])
        cash_balance_vector = np.array([self.cash_balance])
        buy_price_vector = np.array([self.buy_price])
        sell_price_vector = np.array([self.sell_price])

        # Convert 'Date' in the market state to Unix timestamp and then to a numpy array
        if 'Date' in self.market_state:
            self.market_state['Date'] = pd.to_datetime(self.market_state['Date']).timestamp()
        market_state_vector = self.market_state.to_numpy()

        # Convert performance metrics and chosen indicators to numpy arrays
        performance_vector = self.metrics_to_vector(self.performance_metrics)
        indicator_vector = self.indicator_settings_to_vector(self.indicator_settings)

        # Ensure all components have 1 dimension
        if np.ndim(performance_vector) == 0:
            performance_vector = np.array([performance_vector])
        if np.ndim(indicator_vector) == 0:
            indicator_vector = np.array([indicator_vector])

        # Concatenate all components of the state
        full_state = np.concatenate([num_shares_vector, cash_balance_vector, market_state_vector, 
                                    performance_vector, indicator_vector, buy_price_vector, sell_price_vector])

        return full_state

    def initialize_state(self) -> pd.Series:
        """
        This method initializes various aspects of the trading environment such as the portfolio, 
        cash balance, market state, performance metrics, chosen indicators, prices, counters and
        finally concatenates these to form the initial state of the environment.
        """
        # Initialize the portfolio with 0 shares
        self.num_shares: int = 0
        
        # Initialize the current step
        self.current_step: int = 0

        # Initialize the cash balance with the initial cash balance
        self.cash_balance: float = self.initial_cash_balance

        # Initialize the market state with the first row of market data
        self.market_state: pd.Series = self.data.iloc[0]

        # Initialize performance metrics
        self.performance_metrics: Dict[str, float] = self.calculate_initial_metrics()

        # Initialize chosen indicators as an empty dictionary
        self.chosen_indicators: Dict[str, Any] = {}

        # Initialize prices of asset when it was last bought or sold
        self.buy_price: float = 0.0  # Price at which the asset was last bought
        self.sell_price: float = 0.0  # Price at which the asset was last sold

        # Initialize counters for winning trades and total trades
        self.winning_trades: int = 0
        self.total_trades: int = 0

        # Concatenate portfolio state, cash balance, and market state into full state
        self.state: pd.Series = self.concatenate_state()

        return self.state
    
    def metrics_to_vector(self, metrics: Dict[str, Union[float, int]]) -> np.ndarray:
        """
        Converts the given metrics dictionary to a vector (numpy array). 
        The function extracts the values from the dictionary, forms a numpy array, 
        and then checks if all the elements are real numbers (integers or floats).

        Args:
            metrics (Dict[str, Union[float, int]]): The input metrics dictionary.
        
        Raises:
            ValueError: If any element in the created numpy array is not a real number.

        Returns:
            np.ndarray: The resulting numpy array.
        """
        # Convert metrics dictionary to a vector (array), compatible with the rest of the state.
        # This function simply extracts the values and forms a numpy array.
        metrics_vector = np.array([value for value in metrics.values() if np.isreal(value)])

        # Check if all elements are numeric. If not, raise an error.
        if not np.all(np.isreal(metrics_vector)):
            raise ValueError("All elements in metrics_vector should be numeric.")

        return metrics_vector
    
    def step(self, action: Dict[str, Union[str, int]]) -> Tuple[np.ndarray, float, bool]:
        """
        Takes an action based on the current market state and updates the state.

        :param action: A dictionary containing the type of action and relevant details.
        :return: Updated state, calculated reward and a boolean indicating if the episode has ended.
        """
        print("In step")

        # Update the current portfolio value
        self.calculate_portfolio_value() 

        # Calculate the initial reward
        reward = self.calculate_reward()

        # Store the current portfolio value into previous_portfolio_value
        self.previous_portfolio_value = self.current_portfolio_value

        # Handle action
        self.handle_action(action)

        # Get the new market state for the next time step
        self.current_step += 1

        # Check if the episode has ended before getting the next market state
        if self.current_step >= len(self.data):
            done = True
        else:
            self.market_state = self.data.iloc[self.current_step]
            done = False

        # Update performance metrics and state
        self.update_metrics_and_state()

        # Calculate the final reward
        reward = self.calculate_reward()

        return self.state, reward, done

    def handle_action(self, action: Dict[str, Union[str, int]]) -> None:
        """
        Handle the specified action type.

        :param action: A dictionary containing the type of action and relevant details.
        :return: None
        """
        self.last_action = action['type']  # Store the last action

        if action['type'] == 'change_indicator_settings':
            self.update_indicator_settings(action['settings'])

        elif action["type"] == "select_indicators":
            self.select_indicators(action['indicators'])

        elif action['type'] == 'buy':
            self.buy_asset(action['percentage'])

        elif action['type'] == 'sell':
            self.sell_asset(action['percentage'])

        elif action['type'] == 'hold':
            pass  # Nothing to do for hold

    def update_metrics_and_state(self) -> None:
    # Updates performance metrics and state.
    # :return: None

        self.performance_metrics = self.update_metrics()
        self.state = self.concatenate_state()
    
    def check_episode_end(self) -> bool:
        """
        Checks if the episode has ended.

        :return: Boolean indicating if the episode has ended.
        """
        return self.current_step >= len(self.data) - 1 or self.cash_balance <= 0
    
    def reset(self) -> np.ndarray:
        """
        Resets the environment to its initial state and returns the initial state.
        """
        # Reset the environment to the initial state
        self.current_step = 0
        self.portfolio = 0
        self.cash_balance = self.initial_cash_balance
        self.buy_price = 0
        self.sell_price = 0
        self.winning_trades = 0
        self.total_trades = 0

        # Initialize the state
        self.state = self.initialize_state()

        return self.state

    def calculate_reward(self) -> float:
        # Calculate the reward based on the risk-adjusted return, trade penalty, and improvement bonus.

        # Compute the return
        ret = self.compute_return()
        print(f"Return: {ret}")

        # Compute a measure of risk
        risk = self.compute_risk()
        print(f"Risk: {risk}")

        # Compute the risk-adjusted return
        risk_adjusted_return = self.compute_risk_adjusted_return(ret, risk)
        print(f"Risk Adjusted Return: {risk_adjusted_return}")

        # Compute a penalty for trading
        trade_penalty = self.compute_trade_penalty()
        print(f"Trade Penalty: {trade_penalty}")

        # Compute an improvement bonus
        improvement_bonus = self.compute_improvement_bonus(risk_adjusted_return)
        print(f"Improvement Bonus: {improvement_bonus}")

        # The reward is the risk-adjusted return minus the trade penalty plus the improvement bonus
        reward = risk_adjusted_return - trade_penalty + improvement_bonus
        print(f"Reward: {reward}")

        # Store current risk-adjusted return and portfolio value for future comparisons
        self.update_reward_history(risk_adjusted_return)

        return reward
    
    def compute_return(self) -> float:
        """
        Computes the return based on current and previous portfolio value.
        """
        if self.previous_portfolio_value != 0:
            return (self.current_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        else:
            return 0
    
    def compute_risk(self) -> float:
        """
        Computes a measure of risk based on the standard deviation of the portfolio value history.
        """
        risk = np.std(self.portfolio_value_history)
        # Add a small constant to the risk to prevent division by zero
        EPSILON = 1e-8
        return risk + EPSILON

    def compute_risk_adjusted_return(self, ret: float, risk: float) -> float:
        """
        Computes the risk-adjusted return.
        """
        return ret / risk

    def compute_trade_penalty(self) -> float:
        """
        Computes a penalty for trading based on transaction cost and change in portfolio value.
        """
        if self.last_action is None or self.last_action == "hold":
            return 0.0
        else:
            return self.transaction_cost * abs(self.last_trade_value)

    def compute_improvement_bonus(self, risk_adjusted_return: float) -> float:
        """
        Computes an improvement bonus if risk-adjusted return and current portfolio value are 
        both greater than their respective values from lookback_period steps ago.
        """
        improvement_bonus = 0.0
        lookback_period = 10  # Define the period over which improvement is measured

        if self.current_step > lookback_period:
            if len(self.risk_adjusted_return_history) >= lookback_period and len(self.portfolio_value_history) >= lookback_period:
                old_risk_adjusted_return = self.risk_adjusted_return_history[-lookback_period]
                old_portfolio_value = self.portfolio_value_history[-lookback_period]
                if risk_adjusted_return > old_risk_adjusted_return and self.current_portfolio_value > old_portfolio_value:
                    improvement_bonus = 0.1  # This value could be adjusted as per requirements

        return improvement_bonus

    def update_reward_history(self, risk_adjusted_return: float) -> None:
        """
        Updates risk-adjusted return history and portfolio value history.
        """
        self.risk_adjusted_return_history.append(risk_adjusted_return)
        self.portfolio_value_history.append(self.current_portfolio_value)

    def calculate_initial_metrics(self) -> Dict[str, Union[float, int]]:
        """
        This function initializes the performance metrics with default values. 
        The metrics include portfolio value, running average value, drawdown, 
        winning trades, and total trades.

        :return: A dictionary of performance metrics.
        """

        # Initialize with default values
        self.performance_metrics = {
            'Portfolio Value': self.calculate_portfolio_value(),
            'Running Average Value': self.calculate_portfolio_value(),
            'Drawdown': 0,
            'Winning Trades': 0,
            'Total Trades': 0
        }

        # Return the initialized performance metrics
        return self.performance_metrics

    def recalculate_market_data(self) -> pd.DataFrame:
        """
        This function recalculates the market data based on the updated settings 
        of the chosen indicators.

        :return: A pandas DataFrame of recalculated market data.
        """

        # Reset the market data to the original data
        self.market_data = self.original_market_data.copy()

        # Recalculate each indicator in the chosen indicators with the updated settings
        for indicator_name, settings in self.chosen_indicators.items():
            # Get the function for this indicator
            indicator_func = self.INDICATORS[indicator_name]['func']

            # Calculate the indicator and update the market data
            self.market_data[indicator_name] = indicator_func(self.market_data, **settings)

        # Return the recalculated market data
        return self.market_data
    
    def update_metrics(self) -> Dict[str, Union[float, int]]:
        """
        This function updates the performance metrics based on the current state 
        of the portfolio and market.

        :return: A dictionary of updated performance metrics.
        """

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

        # Return the updated performance metrics
        return self.performance_metrics

    def calculate_portfolio_value(self) -> float:
        """
        Function to calculate the current value of the portfolio based on the
        number of shares held and the closing price of the market state.
        """
        portfolio_value = self.market_state['Close'] * self.num_shares
        self.current_portfolio_value = portfolio_value
        return portfolio_value

    def calculate_drawdown(self, current_value: float) -> float:
        """
        Function to calculate the drawdown based on the current value of the 
        portfolio and the historical peak value. Drawdown is a measure of 
        the decline from the historical peak in some variable (usually the 
        cumulative profit or total open equity).
        """
        # Check if historical_peaks attribute exists, if not initialize it with current_value
        if not hasattr(self, 'historical_peaks'):
            self.historical_peaks = current_value
        # Update historical peaks with maximum of current value and existing historical peak value
        self.historical_peaks = max(self.historical_peaks, current_value)
        # Calculate drawdown as a proportion of the decline from the historical peak
        drawdown = (self.historical_peaks - current_value) / self.historical_peaks
        return drawdown
    
    def validate_setting(self, indicator_name: str, param: str, value: Any) -> bool:
        """
        Validate the value of a setting for a given indicator.

        Args:
            indicator_name (str): Name of the indicator.
            param (str): Name of the parameter to validate.
            value (any): Value of the parameter to validate.

        Returns:
            bool: True if the setting is valid, False otherwise.
        """
        param_range = self.INDICATORS[indicator_name]['params'][param]
        if param_range is None:
            return False

        # Check if the value is within the allowed range.
        return param_range.start <= value < param_range.stop

    def update_existing_indicator(self, indicator_name: str, settings: Dict[str, int]) -> None:
        """
        Update the settings for an existing indicator.
        """
        for param, value in settings.items():
            if self.validate_setting(indicator_name, param, value):
                self.chosen_indicators[indicator_name][param] = value
            else:
                raise ValueError(f"Invalid setting: {param} = {value} for indicator: {indicator_name}")

    def add_new_indicator(self, indicator_name: str, settings: Dict[str, int]) -> None:
        """
        Add a new indicator with its settings.
        """
        self.chosen_indicators[indicator_name] = {}
        for param, value in settings.items():
            if self.validate_setting(indicator_name, param, value):
                self.chosen_indicators[indicator_name][param] = value
            else:
                raise ValueError(f"Invalid setting: {param} = {value} for indicator: {indicator_name}")

    def update_indicator_settings(self, new_settings: Dict[str, Dict[str, int]]) -> None:
        """
        Update the settings for the chosen indicators.
        """
        for indicator_name, settings in new_settings.items():
            if indicator_name in self.chosen_indicators:
                for param, value in settings.items():
                    # Validate that the value is within the allowed range for this parameter
                    allowed_range = self.INDICATORS[indicator_name]['params'][param]
                    if allowed_range is not None and (value < allowed_range.start or value >= allowed_range.stop):
                        raise ValueError(f"Invalid value {value} for parameter {param} of indicator {indicator_name}")
                self.update_existing_indicator(indicator_name, settings)
            elif indicator_name in self.INDICATORS:
                self.add_new_indicator(indicator_name, settings)
            else:
                raise ValueError(f"Unknown indicator: {indicator_name}")
        self.calculate_indicators()
    
    def indicator_settings_to_vector(self, settings: Dict[str, Union[int, float]]) -> np.ndarray:
        """
        Convert indicator settings to a vector (numpy array). 
        For simplicity, it is assumed settings are represented by a single scalar value for each indicator.

        Args:
            settings: A dictionary containing the indicator settings.

        Returns:
            A numpy array containing the settings values.

        Raises:
            ValueError: If any element in the settings is not numeric.
        """
        settings_vector = np.array(list(settings.values()))

        # Check if all elements are numeric
        if not np.all(np.isreal(settings_vector)):
            raise ValueError("All elements in settings_vector should be numeric.")

        return settings_vector

    def calculate_indicators(self, params_values: Optional[Dict[str, Dict[str, Any]]]=None) -> None:
        """
        Calculate the value for all chosen indicators.
        This method goes through all the chosen indicators and updates their parameters if necessary.
        It then calculates the indicator values using these parameters and the indicator function, 
        and stores the calculated values.
        
        Args:
            params_values (dict): An optional dictionary containing parameter values for each indicator. 
                                If not provided, the existing parameter values are used. 
        Raises:
            Exception: If market data is not loaded before calling this method.
        """
        # Ensure market data is loaded
        if self.indicator_values is None:
            raise Exception("Market data must be loaded before calculating indicators")
        
        # If no new parameter values provided, use existing ones
        params_values = params_values if params_values is not None else self.params_values

        # Initialize actual_params_values if it doesn't exist
        self.actual_params_values = copy.deepcopy(params_values) if not hasattr(self, 'actual_params_values') else self.actual_params_values

        # Iterate over all chosen indicators
        for indicator_name, settings in self.chosen_indicators.items():
            # Get the function to calculate the indicator
            indicator_func = self.INDICATORS[indicator_name]['func']

            # Get parameters for this indicator from provided values
            params = params_values.get(indicator_name, {}).get('params', {})
            
            # Update the parameters
            params = self.update_parameters(indicator_name, params)

            # Calculate the indicator value with the determined parameters
            self.calculate_and_store_indicator_value(indicator_name, indicator_func, params)
           
    def update_parameters(self, indicator_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the parameters of the indicator by checking if value is a range. 
        If so, a random integer from this range is selected and used as the value for that parameter.

        Args:
            indicator_name (str): Name of the indicator.
            params (dict): Dictionary containing parameter values for the indicator.

        Returns:
            dict: Updated dictionary of parameter values.
        """
        # Create the dictionary for the indicator if it doesn't exist
        if indicator_name not in self.actual_params_values:
            self.actual_params_values[indicator_name] = {}

        for param, value in params.items():
            if isinstance(value, range):
                # Select a random value from the range and store it
                if param not in self.actual_params_values[indicator_name]:
                    self.actual_params_values[indicator_name][param] = random.choice(value)
                params[param] = self.actual_params_values[indicator_name][param]  # use the actual stored value

        return params
    
    def calculate_and_store_indicator_value(self, indicator_name: str, indicator_func: Callable, params: Dict[str, Any]) -> None:
        """
        Calculate and store the indicator value using the provided indicator function and parameters.

        Args:
            indicator_name (str): Name of the indicator.
            indicator_func (callable): Function to calculate the indicator.
            params (dict): Dictionary of parameters to use for the indicator calculation.

        Side Effects:
            If successful, the calculated indicator value is stored in self.indicator_values dictionary.
            If an error occurs during the calculation, the error details are printed to the console.
        """
        try:
            # Calculate the indicator value with the determined parameters
            indicator_value = indicator_func(self.indicator_data, **params)
        except Exception as e:
            print(f"Error calculating indicator {indicator_name} with params {params}")
            print(f"Exception: {e}")
            return
        # Store the calculated indicator value
        self.indicator_values[indicator_name] = indicator_value
                
    def select_indicators(self, chosen_indicators: List[str]) -> None:
        """
        Selects the indicators to be used for trading.

        Args:
            chosen_indicators: A list of indicators to be used.

        Raises:
            ValueError: If an unknown indicator is chosen.
        """
        for indicator_name in chosen_indicators:
            indicator_name = indicator_name.lower()
            if indicator_name not in self.INDICATORS:
                raise ValueError(f"Unknown indicator: {indicator_name}")

        for indicator_name in chosen_indicators:
            indicator_name = indicator_name.lower()
            param_range = self.INDICATORS[indicator_name]['params']['period']
            initial_value = random.randint(param_range.start, param_range.stop - 1)
            self.chosen_indicators[indicator_name] = {'period': initial_value}
   
    def run_episode(self) -> float:
        """
        Runs a single episode of the trading process. 

        Returns:
            The total reward from the episode.
        """
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
    
    def buy_asset(self, percentage: float) -> None:
        """
        Buy an asset given a certain percentage of the current cash balance.

        Args:
            percentage: The percentage of the current cash balance to spend on buying shares.

        Returns:
            None
        """
        cost = self.cash_balance * (percentage / 100)
        amount = cost / self.market_state['Close']

        if cost <= self.cash_balance:
            self.cash_balance -= cost
            self.num_shares += amount
            self.total_trades += 1
                    
    def sell_asset(self, percentage: float) -> None:
        """
        Sell an asset given a certain percentage of the current number of shares.

        Args:
            percentage: The percentage of the current number of shares to sell.

        Returns:
            None
        """
        amount = self.num_shares * (percentage / 100)

        if amount <= self.num_shares:
            proceeds = self.market_state['Close'] * amount
            self.cash_balance += proceeds
            self.num_shares -= amount

            if self.market_state['Close'] > self.buy_price:  
                self.winning_trades += 1

            self.total_trades += 1
            print(f'num_shares after selling: {self.num_shares}')
            print(f'cash_balance after selling: {self.cash_balance}')

    def action_to_vector(self, action: str) -> np.ndarray:
        """
        Convert a given action to a vector.

        Args:
            action: The action to be converted.

        Returns:
            A numpy array representing the action in a binary vector form.
        """
        action_space = ['Buy', 'Sell', 'Hold', 'change_indicator_settings', "select_indicators"]
        action_vector = [0]*len(action_space)
        action_index = action_space.index(action)
        action_vector[action_index] = 1 
        return np.array([action_vector])