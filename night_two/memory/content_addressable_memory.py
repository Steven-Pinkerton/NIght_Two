import time
from typing import Any, List
import numpy as np
import torch
from night_two.memory.memory_unit import MemoryUnit

class ContentAddressableMemoryUnit(MemoryUnit):
    def __init__(self, capacity: int):
        """
        Initialize a memory structure for episodes.

        Args:
            capacity: The maximum number of episodes that can be stored in memory.
        """
        super().__init__()
        self.capacity = capacity
        self.memory = []
        self.content_index = {}

    def write(self, episode: List[np.ndarray]) -> None:
        """
        Write a new episode into memory. If memory is full, remove the oldest episode.

        Args:
            episode: The episode to be written into memory, represented as a list of numpy arrays.
        """
        if len(self.memory) == self.capacity:
            removed_episode = self.memory.pop(0)
            removed_content = self._generate_content_key(removed_episode)
            del self.content_index[removed_content]

        self.memory.append(episode)
        content = self._generate_content_key(episode)
        if content in self.content_index:
            raise ValueError("An episode with the same content already exists in memory")
        self.content_index[content] = episode

    def read(self, index: int) -> List[np.ndarray]:
        """
        Read an episode from memory at the specified index.

        Args:
            index: The index in memory from which to read the episode.

        Returns:
            The episode read from memory at the specified index, represented as a list of numpy arrays.

        Raises:
            IndexError: If the index is out of bounds of the memory.
        """
        if index < 0 or index >= len(self.memory):
            raise IndexError("Memory index out of bounds")
        return self.memory[index]

    def retrieve(self, content: Any) -> List[np.ndarray]:
        """
        Retrieve an episode from memory based on the associated content.

        Args:
            content: The content associated with the episode to retrieve.

        Returns:
            The episode associated with the content, represented as a list of numpy arrays.

        Raises:
            KeyError: If no episode is associated with the content.
        """
        content_key = self._generate_content_key(content)
        if content_key not in self.content_index:
            raise KeyError(f"No episode associated with content {content}")
        return self.content_index[content_key]


    def _generate_content_key(self, content: List[torch.Tensor]) -> str:
        """
        Generate a string representation of the content that can be used as a dictionary key.

        Args:
            content: The content to be converted into a string.

        Returns:
            A string representation of the content.
        """
        flattened_content = [item.view(-1).tolist() for item in content]
        return str(flattened_content)