from typing import List
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM

from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit


class DNC(tf.keras.Model):
    def __init__(self, controller_size: int, memory_size: int, num_read_heads: int, num_write_heads: int):
        super(DNC, self).__init__()
        self.memory_size = memory_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads

        # Initialize LSTM controller with given size
        self.controller = tf.keras.layers.LSTM(controller_size, return_sequences=True)

        # Initialize read and write heads
        self.read_heads = [ReadHead(memory_size) for _ in range(num_read_heads)]
        self.write_heads = [WriteHead(memory_size) for _ in range(num_write_heads)]

        # Initialize memory matrix with zeros and set it as non-trainable
        self.memory = self.add_weight(shape=(self.memory_size, self.memory_size), 
                                      initializer='zeros', trainable=False)

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        """Performs one step of DNC.
        
        Args:
        inputs: Input tensor
        
        Returns:
        A list of read vectors
        """
        # Get controller's output
        controller_output = self.controller(inputs)

        # Each read head reads from the memory matrix
        read_vectors = [head(self.memory, controller_output) for head in self.read_heads]

        # Each write head writes to the memory matrix using the controller's output
        for head in self.write_heads:
            self.memory = head(self.memory, controller_output)

        return read_vectors

class ReadHead(tf.keras.layers.Layer):
    def __init__(self, memory_size: int):
        super(ReadHead, self).__init__()
        self.memory_size = memory_size
        self.key_network = tf.keras.layers.Dense(self.memory_size)

    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        """Reads from the memory matrix.
        
        Args:
        memory: The memory matrix to read from
        controller_output: The output tensor from the controller
        
        Returns:
        A read vector
        """
        # Generate a key from the controller's output
        key = self.key_network(controller_output)
        key = tf.nn.softmax(key, axis=-1)  # normalize the key

        # Calculate the similarities between the key and each memory location
        similarities = tf.keras.losses.cosine_similarity(key, memory)
        read_weights = tf.nn.softmax(similarities, axis=-1)  # normalize the similarities to get read weights

       # Read from the memory matrix based on the read weights
        read_vector = tf.reduce_sum(read_weights[:, tf.newaxis] * memory, axis=1)

        return read_vector

class WriteHead(tf.keras.layers.Layer):
    def __init__(self, memory_size: int):
        super(WriteHead, self).__init__()
        self.memory_size = memory_size
        self.key_network = tf.keras.layers.Dense(self.memory_size)
        self.erase_network = tf.keras.layers.Dense(self.memory_size, activation='sigmoid')
        self.add_network = tf.keras.layers.Dense(self.memory_size)

    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        """Writes to the memory matrix.
        
        Args:
        memory: The memory matrix to write to
        controller_output: The output tensor from the controller
        
        Returns:
        The updated memory matrix
        """
        # Generate a key from the controller's output
        key = self.key_network(controller_output)
        key = tf.nn.softmax(key, axis=-1)  # normalize the key

        # Calculate the similarities between the key and each memory location
        similarities = tf.keras.losses.cosine_similarity(key, memory)
        write_weights = tf.nn.softmax(similarities, axis=-1)  # normalize the similarities to get write weights

        # Generate erase and add vectors
        erase_vector = self.erase_network(controller_output)
        add_vector = self.add_network(controller_output)

        # Update the memory matrix
        memory = (1 - write_weights[:, tf.newaxis] * erase_vector) * memory
        memory += write_weights[:, tf.newaxis] * add_vector

        return memory



class ContentAddressableReadHead(ReadHead):
    def __init__(self, memory_size: int):
        super().__init__(memory_size)

    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        # Generate a key from the controller's output
        key = self.key_network(controller_output)
        key = tf.nn.softmax(key, axis=-1)  # normalize the key

        # expand dimensions of memory
        memory_exp = tf.expand_dims(memory, axis=0)  # now shape is [1, memory_slots, memory_size] 

        # tile memory to match the number of keys in each batch and sequence length
        memory_tiled = tf.tile(memory_exp, [tf.shape(key)[0], tf.shape(key)[1], 1])  # now shape is [batch_size, sequence_length, memory_slots, memory_size] 

        # Calculate the similarities between the key and each memory location
        similarities = tf.keras.losses.cosine_similarity(key, memory_tiled)
        read_weights = tf.nn.softmax(similarities, axis=-1)  # normalize the similarities to get read weights

        # Expand dimensions of memory to match the read_weights shape
        memory_exp = tf.expand_dims(memory, axis=0)

        # Read from the memory matrix based on the read weights
        read_vector = tf.reduce_sum(read_weights[:, tf.newaxis] * memory_exp, axis=1)

        return read_vector

class ContentAddressableWriteHead(WriteHead):
    def __init__(self, memory_size: int):
        super().__init__(memory_size)

    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        # Generate a key from the controller's output
        key = self.key_network(controller_output)
        key = tf.nn.softmax(key, axis=-1)  # normalize the key

        # expand dimensions of memory
        memory_exp = tf.expand_dims(memory, axis=0)  

        # tile memory to match the number of keys in each batch
        memory_tiled = tf.tile(memory_exp, [tf.shape(key)[0], 1, 1])  

        # Calculate the similarities between the key and each memory location
        similarities = tf.keras.losses.cosine_similarity(key, memory_tiled)
        write_weights = tf.nn.softmax(similarities, axis=-1)  # normalize the similarities to get write weights

        # Generate erase and add vectors
        erase_vector = self.erase_network(controller_output)
        add_vector = self.add_network(controller_output)

        # Update the memory matrix
        memory = (1 - write_weights[:, tf.newaxis] * erase_vector) * memory
        memory += write_weights[:, tf.newaxis] * add_vector

        return memory

class ContentAddressableDNC(Model):
    def __init__(self, controller_size=128, memory_size=20, num_read_heads=2, num_write_heads=2, num_memory_slots=100, capacity=100, **kwargs):
        super().__init__(**kwargs)
        self.controller_size = controller_size
        self.memory_size = memory_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.num_memory_slots = num_memory_slots
        self.capacity = capacity

        # Initialize LSTM controller with given size
        self.controller = tf.keras.layers.LSTM(controller_size, return_sequences=True)

        # Initialize read and write heads
        self.read_heads = [ContentAddressableReadHead(memory_size) for _ in range(num_read_heads)]
        self.write_heads = [ContentAddressableWriteHead(memory_size) for _ in range(num_write_heads)]

        # Initialize memory matrix with zeros and set it as non-trainable
        self.memory = self.add_weight(shape=(self.num_memory_slots, self.memory_size), 
                              initializer='zeros', trainable=False)

        # Initialize content addressable memory
        self.content_addressable_memory = ContentAddressableMemoryUnit(capacity)
        
        self.controller = tf.keras.layers.LSTM(controller_size, return_sequences=True)
    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        """Performs one step of DNC.
        
        Args:
        inputs: Input tensor
        
        Returns:
        A list of read vectors
        """
        # Get controller's output
        controller_output = self.controller(inputs)

        # Each read head reads from the memory matrix
        read_vectors = [head(self.memory, controller_output) for head in self.read_heads]

        # Each write head writes to the memory matrix using the controller's output
        for head in self.write_heads:
            self.memory = head(self.memory, controller_output)

        # Write the controller output to the content addressable memory
        self.content_addressable_memory.write(controller_output.numpy().tolist())

        # The output for each timestep is a combination of the controller's output and the read vectors
        output = tf.concat([controller_output] + read_vectors, axis=-1)

        return output