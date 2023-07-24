import numpy as np
from night_two.memory.memory_unit import MemoryUnit


class ContentAddressableMemoryUnit(MemoryUnit):
    def __init__(self, N: int, W: int):
        super().__init__()
        self.N = N  # number of memory cells
        self.W = W  # size of memory vectors
        # Initialize memory as a matrix of zeros with shape (N, W)
        self.memory = np.zeros((N, W))
        self.keys = np.zeros((N, W))  # Store keys separately
        self.counter = 0  # Counter for the next free cell

    def write(self, key: np.ndarray, data: np.ndarray):
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        assert data.shape == (self.W,), "Data must be a 1D numpy array of size W"
        
        # Use a free cell or overwrite the oldest one
        idx = self.counter % self.N

        # Update the corresponding memory cell
        self.memory[idx] = data
        self.keys[idx] = key  # Save the key

        print(f"Wrote data {data} with key {key} at index {idx}")

        self.counter += 1

    def read(self, key: np.ndarray) -> np.ndarray:
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        
        # Find the key with the maximum similarity to the input key
        similarities = np.dot(self.keys, key) / np.sqrt(self.W)
        idx = np.argmax(similarities)

        # Return the corresponding memory cell
        read_data = self.memory[idx]
        print(f"Read data {read_data} with key {self.keys[idx]} at index {idx}")
        return read_data