import numpy as np
from night_two.memory.memory_unit import MemoryUnit



class MemoryMatrix(MemoryUnit):
    def __init__(self, N: int, W: int):
        super().__init__()
        self.N = N  # number of memory cells
        self.W = W  # size of memory vectors
        self.matrix = np.zeros((N, W))  # initialize matrix with zeros
        self.current_address = 0  # pointer to the current memory cell
        self.filled_once = False  # flag to check if memory has been filled at least once

    def write(self, data: np.ndarray) -> None:
        """
        Write data into memory at the current address, and then move the address pointer.

        Args:
            data: The data to be written into memory. Must be a 1D numpy array of size W.
        """
        assert data.shape == (self.W,), "Data must be a 1D numpy array of size W"
        self.matrix[self.current_address] = data

        # Move the address pointer to the next cell, or back to the start if we've reached the end
        self.current_address = (self.current_address + 1) % self.N
        if self.current_address == 0:  # if we have looped back to the start, we have filled the memory once
            self.filled_once = True

    def read(self, address: int) -> np.ndarray:
        """
        Read data from memory at the specified address.

        Args:
            address: The address in memory from which to read data. Must be an integer in [0, N-1].

        Returns:
            The data read from memory at the specified address. A 1D numpy array of size W.
        """
        assert 0 <= address < self.N, "Address must be an integer in [0, N-1]"
        return self.matrix[address]
    
    def has_been_filled(self) -> bool:
        """
        Check whether the memory has been filled up at least once.

        Returns:
            A boolean indicating whether the memory has been filled up at least once.
        """
        return self.filled_once