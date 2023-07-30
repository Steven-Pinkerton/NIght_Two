from typing import List, Optional
import numpy as np
import tensorflow as tf

from night_two.memory.memory_unit import MemoryUnit

class TemporalLinkageMemoryUnit(MemoryUnit):
    def __init__(self, capacity: int):
        """
        Initialize a memory structure for episodes.

        Args:
            capacity: The maximum number of episodes that can be stored in memory.
        """
        super().__init__()
        self.memory = []
        self.capacity = capacity

    def write(self, episode: List[tf.Tensor]) -> None:
        """
        Write a new episode into memory. If memory is full, remove the oldest episode.

        Args:
            episode: The episode to be written into memory, represented as a list of Tensors.
        """
        if not isinstance(episode, list):
            raise TypeError(f"Expected a list, but got {type(episode)}")
        if not all(isinstance(x, tf.Tensor) for x in episode):
            raise TypeError("All items in the episode should be Tensors")

        if len(self.memory) == self.capacity:
            self.memory.pop(0)
        self.memory.append(episode)

    def read(self, index: int) -> List[tf.Tensor]:
        """
        Read an episode from memory at the specified index.

        Args:
            index: The index in memory from which to read the episode.

        Returns:
            The episode read from memory at the specified index, represented as a list of Tensors.

        Raises:
            IndexError: If the index is out of bounds of the memory.
        """
        if index < 0 or index >= len(self.memory):
            raise IndexError("Memory index out of bounds")
        return self.memory[index]

    def predecessor(self, index: int) -> Optional[List[tf.Tensor]]:
        """
        Get the predecessor of the episode at the specified index.

        Args:
            index: The index of the episode whose predecessor to get.

        Returns:
            The predecessor episode, or None if the episode at the specified index is the first episode.
        """
        return self.memory[index - 1] if index > 0 else None

    def successor(self, index: int) -> Optional[List[tf.Tensor]]:
        """
        Get the successor of the episode at the specified index.

        Args:
            index: The index of the episode whose successor to get.

        Returns:
            The successor episode, or None if the episode at the specified index is the last episode.
        """
        return self.memory[index + 1] if index < len(self.memory) - 1 else None

    def save(self, filename: str) -> None:
        """
        Save the memory to a file.

        Args:
            filename: The name of the file to save the memory to.
        """
        filename = filename if filename.endswith(".npy") else filename + ".npy"
        try:
            memory_np = [[tensor.numpy() for tensor in episode] for episode in self.memory]
            np.save(filename, memory_np)
        except Exception as e:
            print(f"An error occurred while saving: {str(e)}")

    def load(self, filename: str) -> None:
        """
        Load the memory from a file.

        Args:
            filename: The name of the file to load the memory from.
        """
        filename = filename if filename.endswith(".npy") else filename + ".npy"
        try:
            self.memory = [[tf.convert_to_tensor(item) for item in episode] for episode in np.load(filename, allow_pickle=True)]
        except Exception as e:
            print(f"An error occurred while loading: {str(e)}")
                
class TemporalLinkageMatrix(tf.Module):
    def __init__(self, num_memory_slots: int):
        super().__init__()
        self.num_memory_slots = num_memory_slots
        self.batch_size = None
        self.linkage_matrix = None  # Initialize with None.

    def update(self, prev_write_weights: tf.Tensor, write_weights: tf.Tensor):
        """Update the linkage matrix based on the write weights."""
        if self.batch_size is None or self.linkage_matrix is None:
            # Infer the batch size from the write weights
            self.batch_size = tf.shape(write_weights)[0]
            # Initialize linkage matrix with zeros and set it as non-trainable
            self.linkage_matrix = tf.zeros(shape=(self.batch_size, self.num_memory_slots, self.num_memory_slots))

        # Calculate the sum of the previous write weights along the second dimension
        sum_prev_write_weights = tf.reduce_sum(prev_write_weights, axis=-1, keepdims=True)

        # Compute a term that captures the fact that once a memory cell is written to,
        # it is no longer the most recent write for any other cell.
        term1 = (1 - sum_prev_write_weights[..., tf.newaxis]) * self.linkage_matrix

        # Here we need to adjust the shape of term2 to match that of term1
        term2 = tf.linalg.matmul(prev_write_weights[..., tf.newaxis], write_weights[..., tf.newaxis, :])
        term2 = tf.reduce_sum(term2, axis=-1, keepdims=True)

        self.linkage_matrix = term1 + term2

    def get(self):
        """Return the current state of the linkage matrix."""
        if self.linkage_matrix is None:
            # If linkage_matrix is None, return None
            return None
        return self.linkage_matrix

    def reset_states(self):
        # Check if batch_size is initialized
        if self.batch_size is not None:
            self.linkage_matrix = tf.zeros(shape=(self.batch_size, self.num_memory_slots, self.num_memory_slots), dtype=tf.float32)
            
            
class ReadTemporalLinkageMatrix(tf.Module):
    def __init__(self, num_memory_slots: int):
        super().__init__()
        self.num_memory_slots = num_memory_slots
        self.batch_size = None
        self.linkage_matrix = None  # type: Optional[tf.Tensor]

    def update(self, prev_read_weights: tf.Tensor, read_weights: tf.Tensor):
        """
        Update the read linkage matrix based on the read weights.
        
        Parameters
        ----------
        prev_read_weights : tf.Tensor
            The read weights from the previous time step.
            Shape: (batch_size, num_memory_slots)
        read_weights : tf.Tensor
            The read weights from the current time step.
            Shape: (batch_size, num_memory_slots)
        """
        if self.batch_size is None or self.linkage_matrix is None:
            self.batch_size = tf.shape(read_weights)[0]
            self.linkage_matrix = tf.zeros(shape=(self.batch_size, self.num_memory_slots, self.num_memory_slots))

        sum_prev_read_weights = tf.reduce_sum(prev_read_weights, axis=-1, keepdims=True)

        term1 = (1 - sum_prev_read_weights[..., tf.newaxis]) * self.linkage_matrix
        term2 = tf.linalg.matmul(prev_read_weights[..., tf.newaxis], read_weights[..., tf.newaxis, :])
        term2 = tf.reduce_sum(term2, axis=-1, keepdims=True)

        self.linkage_matrix = term1 + term2

    def get(self) -> Optional[tf.Tensor]:
        """
        Return the current state of the read linkage matrix.
        
        Returns
        -------
        linkage_matrix : tf.Tensor, optional
            The current state of the read linkage matrix.
            Shape: (batch_size, num_memory_slots, num_memory_slots)
            Returns None if the linkage matrix has not been initialized.
        """
        return self.linkage_matrix

    def reset_states(self):
        """
        Reset the states of the read linkage matrix.
        
        The linkage matrix is reset to zeros.
        """
        if self.batch_size is not None:
            self.linkage_matrix = tf.zeros(shape=(self.batch_size, self.num_memory_slots, self.num_memory_slots), dtype=tf.float32)