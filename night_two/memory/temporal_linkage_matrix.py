class TemporalLinkageMemoryUnit(MemoryUnit):
    def __init__(self):
        self.memory = []
        
    def write(self, data):
        """
        Append data to the end of the memory list.

        Args:
            data: The data to be written into memory.
        """
        self.memory.append(data)

    def read(self, index):
        """
        Read data from memory at the specified index.

        Args:
            index: The index in memory from which to read data.

        Returns:
            The data read from memory at the specified index.
        """
        return self.memory[index] if index < len(self.memory) else None