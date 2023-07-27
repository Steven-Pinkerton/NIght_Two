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

class TestTemporalLinkageMemoryUnit(unittest.TestCase):

    def setUp(self):
        self.capacity = 5
        self.memory_unit = TemporalLinkageMemoryUnit(self.capacity)
        for i in range(self.capacity):
            self.memory_unit.write([np.array([i])])

    def test_capacity(self):
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        self.memory_unit.write([np.array([5])])
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        assert self.memory_unit.memory[0][0][0] == 1

    def test_write(self):
        with self.assertRaises(TypeError):
            self.memory_unit.write("Not a list")
        with self.assertRaises(TypeError):
            self.memory_unit.write([1, 2, 3])

    def test_read(self):
        episode = self.memory_unit.read(2)
        assert episode[0][0] == 2
        with self.assertRaises(IndexError):
            self.memory_unit.read(-1)
        with self.assertRaises(IndexError):
            self.memory_unit.read(self.memory_unit.capacity)

    def test_predecessor_successor(self):
        assert self.memory_unit.predecessor(0) is None
        assert self.memory_unit.successor(self.memory_unit.capacity - 1) is None
        assert self.memory_unit.predecessor(2)[0][0] == 1
        assert self.memory_unit.successor(2)[0][0] == 3

    def test_save_load(self):
        filename = "test_memory.pkl"
        self.memory_unit.save(filename)
        assert os.path.exists(filename)
        new_memory_unit = TemporalLinkageMemoryUnit(self.memory_unit.capacity)
        new_memory_unit.load(filename)
        for i in range(self.memory_unit.capacity):
            assert self.memory_unit.read(i)[0][0] == new_memory_unit.read(i)[0][0]
        os.remove(filename)

class TestContentAddressableMemoryUnit(unittest.TestCase):
    
    def setUp(self):
        self.capacity = 5
        self.memory_unit = ContentAddressableMemoryUnit(self.capacity)
        for i in range(self.capacity):
            self.memory_unit.write([np.array([i])])

    def test_capacity(self):
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        self.memory_unit.write([np.array([5])])
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        assert np.array_equal(self.memory_unit.memory[0][0], np.array([1]))

    def test_write_read(self):
        episode = [np.array([6])]
        self.memory_unit.write(episode)
        read_episode = self.memory_unit.read(self.capacity - 1)
        assert np.array_equal(read_episode[0], episode[0])

    def test_duplicate_content(self):
        episode = [np.array([6])]
        with self.assertRaises(ValueError):
            self.memory_unit.write(episode)
            self.memory_unit.write(episode)  # Duplicate write should raise ValueError

    def test_retrieve(self):
        episode = [np.array([7])]
        self.memory_unit.write(episode)
        retrieved_episode = self.memory_unit.retrieve([np.array([7])])  # Pass np.array([7]) instead of [7]
        assert np.array_equal(retrieved_episode[0], episode[0])
        with self.assertRaises(KeyError):
            self.memory_unit.retrieve([np.array([100])])  # Pass np.array([100]) instead of [100]

if __name__ == '__main__':
    unittest.main()