import unittest
import numpy as np
from night_two.memory.memory_matrix import MemoryMatrix
from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit

class TestMemoryMatrix(unittest.TestCase):

    def test_initialization(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        assert memory.matrix.shape == (N, W)
        assert (memory.matrix == np.zeros((N, W))).all()
        assert memory.current_address == 0
        assert not memory.filled_once
        
    def test_write_and_read(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        data = np.ones(W)
        memory.write(data)
        assert (memory.read(0) == data).all()
        
    def test_write_beyond_size(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        data1 = np.ones(W)
        data2 = np.ones(W) * 2
        for _ in range(N):
            memory.write(data1)
        memory.write(data2)
        assert (memory.read(0) == data2).all()  # the first cell should have been overwritten

    def test_has_been_filled(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        assert not memory.has_been_filled()
        data = np.ones(W)
        for _ in range(N):
            memory.write(data)
        assert memory.has_been_filled()
        # add more test methods here

#class TestContentAddressableMemoryUnit(unittest.TestCase):
 #   def setUp(self):
  #      self.N = 10
   #     self.W = 5
  #      self.memory = ContentAddressableMemoryUnit(self.N, self.W)

  #  def test_initialization(self):
  #      assert self.memory.memory.shape == (self.N, self.W)
  #      assert (self.memory.memory == np.zeros((self.N, self.W))).all()

  #  def test_write_and_read_similar(self):
  #      data1 = np.array([1, 2, 3, 4, 5])
  #      key1 = np.array([0, 0, 0, 0, 0])  # key for data1
  #      print(f"Writing data: {data1} with key: {key1}")
  #      self.memory.write(key1, data1)  # corrected
  #      print(f"Memory after writing: {self.memory.memory}")

  #      data2 = np.array([6, 7, 8, 9, 10])
  #      key2 = np.array([1, 1, 1, 1, 1])  # key for data2
  #      print(f"Writing data: {data2} with key: {key2}")
  #      self.memory.write(key2, data2)  # corrected
  #      print(f"Memory after writing: {self.memory.memory}")

  #      print(f"Reading data with key: {key1}")
  #      read_data1 = self.memory.read(key1)
  #      print(f"Read data: {read_data1}")

  #      print(f"Reading data with key: {key2}")
  #      read_data2 = self.memory.read(key2)
  #      print(f"Read data: {read_data2}")

  #      np.testing.assert_almost_equal(np.linalg.norm(read_data1 - data1), 0, decimal=2)
  #      np.testing.assert_almost_equal(np.linalg.norm(read_data2 - data2), 0, decimal=2)

  #  def test_weighted_average_of_similar_vectors(self):
  #      data1 = np.array([1, 1, 1, 1, 1])
  #      key1 = np.array([0, 0, 0, 0, 0])  # key for data1
  #      self.memory.write(key1, data1)

  #      data2 = np.array([2, 2, 2, 2, 2])
  #      key2 = np.array([1, 1, 1, 1, 1])  # key for data2
  #      self.memory.write(key2, data2)

  #      key = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
  #      read_data = self.memory.read(key)

        # calculate euclidean distances
  #      dist1 = np.linalg.norm(key1 - key)
  #      dist2 = np.linalg.norm(key2 - key)

        # convert distances to weights
  #      weights = [1 / dist1, 1 / dist2]
  #      weights = weights / np.sum(weights)

  #      expected_data = weights[0] * data1 + weights[1] * data2
  #      np.testing.assert_almost_equal(read_data, expected_data, decimal=6)

  #  def test_multiple_writes_and_reads(self):
  #      data1 = np.array([1, 2, 3, 4, 5])
  #      data2 = np.array([6, 7, 8, 9, 10])
  #      key1 = np.array([1, 2, 3, 4, 5])  # Keys to use
  #      key2 = np.array([6, 7, 8, 9, 10])  # Keys to use
  #      self.memory.write(key1, data1)  # Use keys to write data
  #      self.memory.write(key2, data2)  # Use keys to write data
  #      read_data1 = self.memory.read(key1)
  #      read_data2 = self.memory.read(key2)
  #      np.testing.assert_almost_equal(np.linalg.norm(read_data1 - data1), 0, decimal=2)
  #      np.testing.assert_almost_equal(np.linalg.norm(read_data2 - data2), 0, decimal=2)
        
  #  def test_overwrite_memory(self):
  #      data1 = np.array([1, 2, 3, 4, 5])
  #      key1 = np.array([0, 0, 0, 0, 0])  # key for data1
  #      self.memory.write(key1, data1)  # corrected

 #       data2 = np.array([6, 7, 8, 9, 10])
 #       key2 = np.array([1, 1, 1, 1, 1])  # key for data2
 #       self.memory.write(key2, data2)  # corrected

 #       read_data = self.memory.read(key2)
 #       np.testing.assert_almost_equal(read_data, data2, decimal=6)
        
 #   def test_different_keys(self):
  #      key = np.array([1, 2, 3, 4, 5])
   #     data = np.zeros(self.memory.W)  # Use a numpy array of zeros as data
   #     self.memory.write(key, data)
     #   read_data = self.memory.read(key)
    #    print(read_data)  # Print read data for debugging
   #     print(data)  # Print expected data for debugging
  #      np.testing.assert_almost_equal(read_data, data, decimal=2)
        
if __name__ == '__main__':
    unittest.main()