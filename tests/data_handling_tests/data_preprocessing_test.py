import os
import pandas as pd
import numpy as np
import unittest
from night_two.data_handling.data_preprocessing import preprocess_data  # replace "your_file_name" with the actual name of your python file

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a test CSV file
        self.test_filename = 'test_preprocess_data.csv'
        data = pd.DataFrame({
            "Open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            "Close": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            }, index=pd.date_range('2022-01-01', periods=10)
            )
        data.to_csv(self.test_filename)

    def tearDown(self):
        # Remove test CSV files
        os.remove(self.test_filename)
        os.remove('preprocessed_' + self.test_filename)

    def test_preprocess_data(self):
        preprocessed_data = preprocess_data(self.test_filename)

        # Check that the returned DataFrame has the correct columns
        self.assertTrue(set(['Open', 'Close']).issubset(preprocessed_data.columns))

        # Check that missing values have been filled
        self.assertFalse(preprocessed_data.isnull().any().any())

        # Check that the data has been standardized
        self.assertTrue(np.allclose(preprocessed_data.mean(), 0))
        self.assertTrue(np.all(np.isclose(preprocessed_data.std(), 1, rtol=1e-05)))  # Updated line

        # Check that the preprocessed file exists
        self.assertTrue(os.path.exists('preprocessed_' + self.test_filename))

        # Check the contents of the file
        saved_data = pd.read_csv('preprocessed_' + self.test_filename, index_col=0)
        pd.testing.assert_frame_equal(saved_data, preprocessed_data)

if __name__ == "__main__":
    unittest.main()