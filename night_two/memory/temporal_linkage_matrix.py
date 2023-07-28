from typing import List, Optional

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
        tf.saved_model.save(self.memory, filename)

    def load(self, filename: str) -> None:
        """
        Load the memory from a file.

        Args:
            filename: The name of the file to load the memory from.
        """
        self.memory = tf.saved_model.load(filename)