import os
import pandas as pd
import unittest
from data_download import download_data, save_data  # replace "your_file_name" with the actual name of your python file

class TestDataDownload(unittest.TestCase):

    def test_download_data_valid(self):
        data = download_data('^RUT', '2022-01-01', '2022-01-31', period='1d')
        self.assertIsNotNone(data)
        self.assertTrue(set(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']).issubset(data.columns))

    def test_download_data_invalid(self):
        data = download_data('INVALID', '2022-01-01', '2022-01-31', period='1d')
        self.assertIsNone(data)

    def test_save_data(self):
        test_filename = 'test_save_data.csv'
        
        # Check the file does not exist before the save operation
        self.assertFalse(os.path.exists(test_filename))
        
        # Save data to a file
        data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        save_data(data, test_filename)
        
        # Check the file exists after the save operation
        self.assertTrue(os.path.exists(test_filename))
        
        # Check the contents of the file
        saved_data = pd.read_csv(test_filename, index_col=0)
        pd.testing.assert_frame_equal(saved_data, data)
        
        # Clean up by removing the file created during the test
        os.remove(test_filename)


if __name__ == "__main__":
    unittest.main()
