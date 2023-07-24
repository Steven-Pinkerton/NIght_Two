import unittest
import numpy as np
from night_two.memory.memory_unit import MemoryUnit
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

class ContentAddressableMemoryUnit(MemoryUnit):
    def __init__(self, N: int, W: int):
        super().__init__()
        self.N = N  # number of memory cells
        self.W = W  # size of memory vectors
        # Initialize memory as a matrix of zeros with shape (N, W)
        self.memory = np.zeros((N, W))  
        self.keys = np.zeros((N, W))  # Store keys separately
        self.usage = np.zeros(N)  # Keep track of usage for each cell

    def write(self, key: np.ndarray, data: np.ndarray):
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        assert data.shape == (self.W,), "Data must be a 1D numpy array of size W"
        
        # Calculate cosine similarities
        similarities = np.dot(self.keys, key) / (np.linalg.norm(self.keys, axis=1) * np.linalg.norm(key))
        idx = np.argmax(similarities)

        # If maximum similarity is not 1, find least recently used cell
        if similarities[idx] < 1:
            idx = np.argmin(self.usage)

        # Update the corresponding memory cell
        self.memory[idx] = data
        self.keys[idx] = key  # Save the key
        self.usage += 1  # Increase usage count for all cells
        self.usage[idx] = 0  # Reset usage count for current cell

    def read(self, key: np.ndarray) -> np.ndarray:
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        
        # Calculate cosine similarities
        similarities = np.dot(self.keys, key) / (np.linalg.norm(self.keys, axis=1) * np.linalg.norm(key))
        idx = np.argmax(similarities)

        # Increase usage count for the read cell
        self.usage[idx] += 1

        # Return the corresponding memory cell
        return self.memory[idx]
        
if __name__ == '__main__':
    unittest.main()