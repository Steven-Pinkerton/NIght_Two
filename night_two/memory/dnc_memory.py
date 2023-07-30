from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM
from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit
from night_two.memory.temporal_linkage_matrix import TemporalLinkageMatrix

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
    def __init__(self, memory_size: int, num_memory_slots: int):
        super().__init__(memory_size)
        self.num_memory_slots = num_memory_slots 

    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        # Generate a key from the controller's output
        key = self.key_network(controller_output)
        key = tf.nn.softmax(key, axis=-1)    # normalize the key

        # reshape memory to match the key size
        memory_reshaped = tf.reshape(memory, [1, 1, self.num_memory_slots, self.memory_size])

        # tile memory to match the number of keys in each batch and sequence length
        memory_tiled = tf.tile(memory_reshaped, [tf.shape(controller_output)[0], tf.shape(controller_output)[1], 1, 1])

        # Calculate the similarities between the key and each memory location
        similarities = tf.keras.losses.cosine_similarity(key[..., tf.newaxis, :], memory_tiled)
        read_weights = tf.nn.softmax(similarities, axis=-1)  # normalize the similarities to get read weights

        # Read from the memory matrix based on the read weights
        read_vector = tf.reduce_sum(read_weights[..., tf.newaxis] * memory_tiled, axis=2)

        return read_vector

class ContentAddressableWriteHead(WriteHead):
    def __init__(self, memory_size: int, num_memory_slots: int):
        super().__init__(memory_size)
        self.num_memory_slots = num_memory_slots

        # Initialize the networks to produce outputs of the correct size
        self.key_network = tf.keras.layers.Dense(self.memory_size)
        self.erase_network = tf.keras.layers.Dense(self.memory_size, activation='sigmoid')
        self.add_network = tf.keras.layers.Dense(self.memory_size, activation='relu')

    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        # Generate a key from the controller's output
        key = self.key_network(controller_output)
        key = tf.nn.softmax(key, axis=-1)

        # reshape memory to match the key size
        memory_reshaped = tf.reshape(memory, [1, 1, self.num_memory_slots, self.memory_size])

        # tile memory to match the number of keys in each batch and sequence length
        memory_tiled = tf.tile(memory_reshaped, [tf.shape(controller_output)[0], tf.shape(controller_output)[1], 1, 1])

        # Calculate the similarities between the key and each memory location
        similarities = tf.keras.losses.cosine_similarity(key[..., tf.newaxis, :], memory_tiled)
        write_weights = tf.nn.softmax(similarities, axis=-1)  # normalize the similarities to get write weights

        # Generate erase and add vectors
        erase_vector = self.erase_network(controller_output)
        add_vector = self.add_network(controller_output)

        # Expand dimensions of erase_vector and add_vector to match the write_weights
        erase_vector = tf.expand_dims(erase_vector, axis=-2)  # shape now: [batch_size, sequence_length, 1, memory_size]
        add_vector = tf.expand_dims(add_vector, axis=-2)  # shape now: [batch_size, sequence_length, 1, memory_size]

        # Tile erase_vector and add_vector across the third dimension to match the number of memory slots
        erase_vector = tf.tile(erase_vector, [1, 1, self.num_memory_slots, 1])
        add_vector = tf.tile(add_vector, [1, 1, self.num_memory_slots, 1])

        # Expand dimensions of write_weights before multiplication
        write_weights_expanded = tf.expand_dims(write_weights, axis=-1)

        # Update the memory matrix
        memory_tiled = (1 - write_weights_expanded * erase_vector) * memory_tiled + write_weights_expanded * add_vector
        
        # Aggregate memory_tiled across the batch size and sequence length dimensions to update self.memory
        self.memory = tf.reduce_mean(tf.reduce_mean(memory_tiled, axis=0), axis=0)

        return self.memory

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
         
        # Initialize read and write heads with memory size and number of memory slots
        self.read_heads = [ContentAddressableReadHead(memory_size, num_memory_slots) for _ in range(num_read_heads)]
        self.write_heads = [ContentAddressableWriteHead(memory_size, num_memory_slots) for _ in range(num_write_heads)]


        # Initialize memory matrix with zeros and set it as non-trainable
        self.memory = self.add_weight(shape=(self.num_memory_slots, self.memory_size), 
                              initializer='zeros', trainable=False)

        # Initialize content addressable memory
        self.content_addressable_memory = ContentAddressableMemoryUnit(capacity)
    
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

class ContentAddressableWriteHeadWithLinkage(tf.Module):
    def __init__(self, memory_size: int, num_memory_slots: int):
        super().__init__()
        self.memory_size = memory_size
        self.num_memory_slots = num_memory_slots

        # Initialize the temporal linkage matrix
        self.temporal_linkage_matrix = TemporalLinkageMatrix(num_memory_slots)

        # Initialize the previous write weights as zeros
        self.prev_write_weights = tf.zeros(shape=(num_memory_slots,), dtype=tf.float32)

        self.reset_states()

    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        # Infer batch size from the controller output
        self.batch_size = tf.shape(controller_output)[0]
        
        # Initialize the previous write weights here
        self.prev_write_weights = tf.zeros(shape=(self.batch_size, self.num_memory_slots,), dtype=tf.float32)

        key, erase_vector, write_vector = self._parse_controller_output(controller_output)

        # Reshape the key to match the shape of memory
        key = tf.reshape(key, [-1, 1, self.memory_size])

        similarities = tf.keras.losses.cosine_similarity(key, memory)
        self.write_weights = tf.nn.softmax(similarities, axis=1)

        erase_vector_broadcasted = tf.broadcast_to(erase_vector[:, tf.newaxis, :], [self.batch_size, self.num_memory_slots, self.memory_size])
        erase_term = tf.einsum('ij,ijk->ijk', self.write_weights, erase_vector_broadcasted)
        write_vector_broadcasted = tf.broadcast_to(write_vector[:, tf.newaxis, :], [self.batch_size, self.num_memory_slots, self.memory_size])
        write_term = tf.einsum('ij,ijk->ijk', self.write_weights, write_vector_broadcasted)

        next_memory_state = memory * (1 - erase_term) + write_term

        # Update the temporal linkage matrix based on the previous and current write weights
        self.temporal_linkage_matrix.update(self.prev_write_weights, self.write_weights)

        # Update the previous write weights with the current write weights
        self.prev_write_weights = self.write_weights

        return next_memory_state

    def reset_states(self):
        # Reset the previous write weights and the temporal linkage matrix when resetting the states of the head
        self.prev_write_weights = tf.zeros(shape=(self.num_memory_slots,), dtype=tf.float32)
        self.temporal_linkage_matrix.reset_states()

    def get_write_weights(self):
        return self.write_weights

    def get_temporal_linkage_matrix(self):
        return self.temporal_linkage_matrix.get()

    def _parse_controller_output(self, controller_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        key = controller_output[:, :self.memory_size]
        erase_vector = tf.sigmoid(controller_output[:, self.memory_size:self.memory_size*2])
        write_vector = controller_output[:, self.memory_size*2:self.memory_size*3]
        return key, erase_vector, write_vector

    def get_prev_write_weights(self):
        return self.prev_write_weights