import pandas as pd
import numpy as np
import night_two.data_handling.indicators

def test_indicators_mapping():
    data = pd.DataFrame({
        'High': np.random.rand(50) * 100,
        'Low': np.random.rand(50) * 100,
        'Close': np.random.rand(50) * 100,
        'Volume': np.random.randint(100, 10000, 50),
    })

    for indicator_name, indicator_info in night_two.data_handling.indicators.INDICATORS.items():
        func = indicator_info['func']
        params = indicator_info['params']

        if indicator_name in ['sma', 'rsi']:
            period = 14
            assert not func(data, period).isna().any(), f"{indicator_name} calculation resulted in NaN"

        # Add more conditions for different indicators depending on their required parameters.

test_indicators_mapping()