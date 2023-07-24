import time
import numpy as np
from night_two.memory.memory_unit import MemoryUnit


class ContentAddressableMemoryUnit(MemoryUnit):
    def __init__(self, N: int, W: int):
        super().__init__()
        self.N = N  # number of memory cells
        self.W = W  # size of memory vectors
        self.memory = np.zeros((N, W))  
        self.keys = np.zeros((N, W))
        self.usage = np.full(N, np.inf)  # Unused cells have usage set to infinity
        self.timestamp = np.zeros(N)  # New attribute to store timestamps

    def write(self, key: np.ndarray, data: np.ndarray):
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        assert data.shape == (self.W,), "Data must be a 1D numpy array of size W"

        # Find the index with the exact same key or the least recently used one
        match_idx = np.where(np.all(self.keys == key, axis=1))
        if match_idx[0].size > 0:
            idx = match_idx[0][0]
        else:
            idx = np.argmin(self.timestamp)

        self.keys[idx] = key
        self.memory[idx] = data
        self.usage[idx] = 0
        self.timestamp[idx] = time.time()  # Update the timestamp
        self.usage += 1

    def read(self, key: np.ndarray) -> np.ndarray:
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"

        # Find the index with the exact same key or the most recently used one
        match_idx = np.where(np.all(self.keys == key, axis=1))
        if match_idx[0].size > 0:
            idx = match_idx[0][0]
        else:
            idx = np.argmax(self.timestamp)

        self.usage[idx] = 0
        self.timestamp[idx] = time.time()  # Update the timestamp
        self.usage += 1

        return self.memory[idx]