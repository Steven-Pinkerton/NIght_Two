import numpy as np
from night_two.memory.memory_unit import MemoryUnit


class ContentAddressableMemoryUnit(MemoryUnit):
    def __init__(self, N: int, W: int):
        super().__init__()
        self.N = N  # number of memory cells
        self.W = W  # size of memory vectors
        # Initialize memory as a matrix of zeros with shape (N, W)
        self.memory = np.zeros((N, W))  

    def write(self, key: np.ndarray, data: np.ndarray):
        if not isinstance(key, np.ndarray):
            raise TypeError(f"Key must be a numpy array, but got {type(key).__name__}")
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data must be a numpy array, but got {type(data).__name__}")
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        assert data.shape == (self.W,), "Data must be a 1D numpy array of size W"
        similarities = np.dot(self.memory, key) / np.sqrt(self.W) + 1e-9  # Add small constant
        weights = np.exp(similarities) / np.sum(np.exp(similarities))  # Get weights
        print(weights)  # Print weights for debugging
        self.memory = 1 / (1 + np.exp(-self.memory + np.outer(weights, data)))  # Use sigmoid instead of tanh
        return weights

    def read(self, key: np.ndarray) -> np.ndarray:
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        similarities = np.dot(self.memory, key) / np.sqrt(self.W) + 1e-9  # Add small constant
        weights = np.exp(similarities) / np.sum(np.exp(similarities))
        print(weights)  # Print weights for debugging
        return np.dot(weights, self.memory)