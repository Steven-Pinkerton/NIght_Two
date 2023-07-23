import numpy as np
from night_two.memory.memory_unit import MemoryUnit


class ContentAddressableMemoryUnit(MemoryUnit):
    def __init__(self, N: int, W: int):
        super().__init__()
        self.N = N  # number of memory cells
        self.W = W  # size of memory vectors
        # Initialize memory as a matrix of zeros with shape (N, W)
        self.memory = np.zeros((N, W))  

    def write(self, data: np.ndarray, address: int) -> None:
        """
        Write data into memory at the specified address.

        Args:
            data: The data to be written into memory. It must be a 1D numpy array of size W.
            address: The integer address at which to write the data.
        """
        # Check that the data is the right shape
        assert data.shape == (self.W,), "Data must be a 1D numpy array of size W"
        # Write the data to the specified address
        self.memory[address] = data

    def read(self, key: np.ndarray) -> np.ndarray:
        """
        Read data from memory based on the content.

        Args:
            key: The query key. It must be a 1D numpy array of size W.

        Returns:
            A 1D numpy array of size W, which represents the data read from memory based on the key.
        """
        # Check that the key is the right shape
        assert key.shape == (self.W,), "Key must be a 1D numpy array of size W"
        # Compute the L2 norm of the key
        key_norm = np.linalg.norm(key) + 1e-10  # Add small constant to prevent division by zero
        # Normalize the key by dividing each component by the norm
        key = key / key_norm
        # Compute the L2 norms of the memory vectors
        memory_norm = np.linalg.norm(self.memory, axis=1) + 1e-10  # Normalize, prevent division by zero
        # Compute cosine similarities between the key and each memory cell by taking the dot product
        # of the normalized memory vectors and the normalized key
        similarities = np.dot(self.memory / memory_norm[:, None], key)
        # Compute softmax weights based on the similarities
        weights = np.exp(similarities) / np.sum(np.exp(similarities))
        # Return a weighted sum of memory contents as the output, which is a form of content-based addressing
        return np.dot(weights, self.memory)