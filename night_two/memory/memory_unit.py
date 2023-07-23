import numpy as np


class MemoryUnit:
    def __init__(self):
          if type(self) is MemoryUnit:
            raise NotImplementedError("MemoryUnit is a base class and should not be instantiated directly")

    def write(self, data):
        """
        Write data into memory.

        Args:
            data: The data to be written into memory.
        """
        raise NotImplementedError("The write method should be implemented by subclasses")

    def read(self, address):
        """
        Read data from memory.

        Args:
            address: The address in memory from which to read data.

        Returns:
            The data read from memory.
        """
        raise NotImplementedError("The read method should be implemented by subclasses")

class ContentAddressableMemoryUnit(MemoryUnit):
    def __init__(self):
        self.memory = {}

    def write(self, data, address):
        """
        Write data into memory at the specified address.

        Args:
            data: The data to be written into memory.
            address: The address at which to write the data.
        """
        self.memory[address] = data

    def read(self, address):
        """
        Read data from memory at the specified address.

        Args:
            address: The address in memory from which to read data.

        Returns:
            The data read from memory at the specified address.
        """
        return self.memory.get(address)

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

class ReadHead(MemoryUnit):
    def __init__(self):
        super().__init__()

    def write(self, data):
        pass

    def read(self, address):
        pass

class WriteHead(MemoryUnit):
    def __init__(self):
        super().__init__()

    def write(self, data):
        pass

    def read(self, address):
        pass

class UsageVector:
    def __init__(self):
        self.vector = None

class TemporalLinkageMatrix:
    def __init__(self):
        self.matrix = None

class Controller:
    def __init__(self):
        self.interface_vector = None

    def update_interface_vector(self, data):
        pass

class DNC:
    def __init__(self):
        self.memory_matrix = MemoryMatrix()
        self.read_head = ReadHead()
        self.write_head = WriteHead()
        self.usage_vector = UsageVector()
        self.temporal_linkage_matrix = TemporalLinkageMatrix()
        self.controller = Controller()

    def step(self, input_data):
        pass