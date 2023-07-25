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

        match_idx = np.where(np.all(np.isclose(self.keys, key), axis=1))
        if match_idx[0].size > 0:
            idx = match_idx[0][0]
        else:
            idx = np.argmin(self.timestamp)  # Here, we're prioritizing overwriting the oldest cell

        self.keys[idx] = key
        self.memory[idx] = data
        self.usage[idx] = 0
        self.timestamp[idx] = time.time()  # Here, we're updating the timestamp

        print(f"Writing data {data} with key {key} at index {idx}")

    def read(self, key: np.ndarray) -> np.ndarray:
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"

        print(f"Reading data for key: {key}")

        # Here, we're computing the Euclidean distances between the provided key and all the keys in memory
        euclidean_distances = np.linalg.norm(self.keys - key, axis=1)

        # Here, we're computing similarities as inverses of non-zero Euclidean distances
        similarities = np.zeros(self.N)
        non_zero_indices = euclidean_distances != 0
        zero_indices = euclidean_distances == 0
        similarities[non_zero_indices] = 1 / euclidean_distances[non_zero_indices]
        similarities[zero_indices] = np.inf

        # If exact match found, return that value directly
        if np.any(zero_indices):
            idx = np.where(zero_indices)[0][0]
            print(f"Exact match found: {self.memory[idx]}")
            return self.memory[idx]
        else:
            # Here, we're normalizing the similarities to sum to 1
            total = np.sum(similarities)
            epsilon = 1e-10
            weights = similarities / (total + epsilon)
            print(f"Weights: {weights}")

            # Here, we're computing the weighted average
            weighted_average = np.average(self.memory, axis=0, weights=weights) 
            print(f"Weighted average: {weighted_average}")

            return weighted_average