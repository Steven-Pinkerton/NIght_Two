import os
import unittest
import numpy as np
import pytest
from night_two.memory.memory_matrix import MemoryMatrix
from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit
from night_two.memory.temporal_linkage_matrix import TemporalLinkageMemoryUnit

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
  
class TestTemporalLinkageMemoryUnit:
    @pytest.fixture
    def setup_memory_unit(self):
        capacity = 5
        memory_unit = TemporalLinkageMemoryUnit(capacity)
        for i in range(capacity):
            memory_unit.write([np.array([i])])
        return memory_unit

    def test_capacity(self, setup_memory_unit):
        assert len(setup_memory_unit.memory) == setup_memory_unit.capacity
        setup_memory_unit.write([np.array([5])])
        assert len(setup_memory_unit.memory) == setup_memory_unit.capacity
        assert setup_memory_unit.memory[0][0][0] == 1

    def test_write(self, setup_memory_unit):
        with pytest.raises(TypeError):
            setup_memory_unit.write("Not a list")
        with pytest.raises(TypeError):
            setup_memory_unit.write([1, 2, 3])

    def test_read(self, setup_memory_unit):
        episode = setup_memory_unit.read(2)
        assert episode[0][0] == 2
        with pytest.raises(IndexError):
            setup_memory_unit.read(-1)
        with pytest.raises(IndexError):
            setup_memory_unit.read(setup_memory_unit.capacity)

    def test_predecessor_successor(self, setup_memory_unit):
        assert setup_memory_unit.predecessor(0) is None
        assert setup_memory_unit.successor(setup_memory_unit.capacity - 1) is None
        assert setup_memory_unit.predecessor(2)[0][0] == 1
        assert setup_memory_unit.successor(2)[0][0] == 3

    def test_save_load(self, setup_memory_unit):
        filename = "test_memory.pkl"
        setup_memory_unit.save(filename)
        assert os.path.exists(filename)
        new_memory_unit = TemporalLinkageMemoryUnit(setup_memory_unit.capacity)
        new_memory_unit.load(filename)
        for i in range(setup_memory_unit.capacity):
            assert setup_memory_unit.read(i)[0][0] == new_memory_unit.read(i)[0][0]
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()