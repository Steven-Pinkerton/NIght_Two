from typing import List, Optional, Tuple
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM
from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit
from night_two.memory.temporal_linkage_matrix import ReadTemporalLinkageMatrix, TemporalLinkageMatrix

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
    def __init__(self, memory_size: int, num_memory_slots: int, batch_size: int):
        super().__init__()
        self.memory_size = memory_size
        print(f"memory_size: {self.memory_size}")  
        self.num_memory_slots = num_memory_slots

        self.temporal_linkage_matrix = TemporalLinkageMatrix(num_memory_slots)

        # Initializing the previous write weights with the batch size
        self.prev_write_weights = tf.zeros(shape=(batch_size, num_memory_slots,), dtype=tf.float32)

        self.reset_states()


    def call(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(controller_output)[0]  # <-- Add this line
        key, strength, erase_vector, write_vector = self._parse_controller_output(controller_output)

        num_elements = tf.size(key).numpy()  # Check number of elements in the tensor
        print(f"Number of elements in 'key': {num_elements}")  # Print the number of elements for debugging

        key = tf.reshape(key, [batch_size, 1, self.memory_size])

        similarities = 1 - tf.keras.losses.cosine_similarity(key, memory)
        self.write_weights = tf.nn.softmax(similarities, axis=1)

        erase_vector_broadcasted = tf.broadcast_to(erase_vector[:, tf.newaxis, :], [tf.shape(self.prev_write_weights)[0], self.num_memory_slots, self.memory_size])
        erase_term = tf.einsum('ij,ijk->ijk', self.write_weights, erase_vector_broadcasted)
        write_vector_broadcasted = tf.broadcast_to(write_vector[:, tf.newaxis, :], [tf.shape(self.prev_write_weights)[0], self.num_memory_slots, self.memory_size])
        write_term = tf.einsum('ij,ijk->ijk', self.write_weights, write_vector_broadcasted)

        next_memory_state = memory * (1 - erase_term) + write_term
        
        # Update the temporal linkage matrix based on the previous and current write weights
        self.temporal_linkage_matrix.update(self.prev_write_weights, self.write_weights)

        # Update the previous write weights with the current write weights
        self.prev_write_weights = self.write_weights

        return next_memory_state

    def reset_states(self):
        # Reset the temporal linkage matrix when resetting the states of the head
        self.temporal_linkage_matrix.reset_states()
        self.prev_write_weights = tf.zeros_like(self.prev_write_weights)

    def get_write_weights(self):
        return self.write_weights

    def get_temporal_linkage_matrix(self):
        return self.temporal_linkage_matrix.get()

    def _parse_controller_output(self, controller_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        print(f"controller_output shape: {controller_output.shape}")  # debug line
        assert controller_output.shape[1] >= 4*self.memory_size, f"controller_output shape {controller_output.shape} is incompatible with memory_size {self.memory_size}"
        key = controller_output[:, :self.memory_size]
        print(f"key shape: {key.shape}")  # debug line
        strength = tf.nn.softplus(controller_output[:, self.memory_size:2*self.memory_size])
        print(f"strength shape: {strength.shape}")  # debug line
        erase_vector = tf.nn.sigmoid(controller_output[:, 2*self.memory_size:3*self.memory_size])
        print(f"erase_vector shape: {erase_vector.shape}")  # debug line
        write_vector = tf.nn.sigmoid(controller_output[:, 3*self.memory_size:])
        print(f"write_vector shape: {write_vector.shape}")  # debug line
        return key, strength, erase_vector, write_vector


    def get_prev_write_weights(self):
        return self.prev_write_weights
    
class ContentAddressableReadHeadWithLinkage(tf.Module):
    def __init__(self, memory_size: int, num_memory_slots: int, batch_size: int):
        super().__init__()
        self.memory_size = memory_size
        print(f"memory_size: {self.memory_size}")  
        self.num_memory_slots = num_memory_slots
        self.read_temporal_linkage_matrix = ReadTemporalLinkageMatrix(num_memory_slots)

        # Initialize the previous read weights with batch size
        self.prev_read_weights = tf.zeros(shape=(batch_size, num_memory_slots,), dtype=tf.float32)

        self.reset_states()

    def __call__(self, memory: tf.Tensor, controller_output: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(controller_output)[0]  # <-- Add this line
        key, strength = self._parse_controller_output(controller_output)

        num_elements = tf.size(key).numpy()  # Check number of elements in the tensor
        print(f"Number of elements in 'key': {num_elements}")  # Print the number of elements for debugging

        print(f'key.shape: {key.shape}, expected shape: ({batch_size}, {self.memory_size}), actual number of elements: {num_elements}, expected number of elements: {batch_size * self.memory_size}')
        assert num_elements == batch_size * self.memory_size, "Mismatch between the number of elements in 'key' and the expected number."


        key = tf.reshape(key, [batch_size, 1, self.memory_size])
        similarities = 1 - tf.keras.losses.cosine_similarity(key, memory)
        self.read_weights = tf.nn.softmax(strength * similarities, axis=1)


        read_vectors = tf.einsum('ij,ijk->ik', self.read_weights, memory)

        self.read_temporal_linkage_matrix.update(self.prev_read_weights, self.read_weights)

        self.prev_read_weights = self.read_weights

        return read_vectors

    def reset_states(self):
        self.read_temporal_linkage_matrix.reset_states()
        self.prev_read_weights = tf.zeros_like(self.prev_read_weights)

    def get_read_weights(self) -> tf.Tensor:
        return self.read_weights

    def get_temporal_linkage_matrix(self) -> Optional[tf.Tensor]:
        return self.read_temporal_linkage_matrix.get()

    def _parse_controller_output(self, controller_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        print(f"controller_output shape: {controller_output.shape}")  # debug line
        key = controller_output[:, :self.memory_size]
        print(f"key shape: {key.shape}")  # debug line
        strength = tf.nn.softplus(controller_output[:, self.memory_size:2*self.memory_size])
        return key, strength

    def get_prev_read_weights(self) -> Optional[tf.Tensor]:
        return self.prev_read_weights

class Controller(tf.keras.Model):
    def __init__(self, hidden_size):
        super(Controller, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)

    def call(self, inputs, states):
        return self.lstm(inputs, initial_state=states)

    def get_initial_state(self, inputs=None, batch_size=None):
        return self.lstm.get_initial_state(inputs)

DNCState = namedtuple('DNCState', [
    'memory', 
    'read_vectors', 
    'read_weights', 
    'write_weights', 
    'linkage_matrix', 
    'precedence_weights', 
    'controller_state'
])

class DNCTwo(tf.Module):
    def __init__(self, input_dim, num_memory_slots, memory_vector_dim, num_read_heads, controller_hidden_size):
        super().__init__()
        # Parameters
        self.input_dim = input_dim
        self.num_memory_slots = num_memory_slots
        self.memory_vector_dim = memory_vector_dim
        self.num_read_heads = num_read_heads
        self.controller_hidden_size = controller_hidden_size
        

        # Controller
        self.controller = Controller(self.controller_hidden_size)

        
        # Memory
        self.memory = Memory(self.num_memory_slots, self.memory_vector_dim)

        # Temporal Linkage Matrix
        self.temporal_linkage_matrix = TemporalLinkageMatrix(self.num_memory_slots)

        # Interface parameters for read/write heads
        # Key vectors for read heads
        self.read_keys = tf.keras.layers.Dense(units=self.num_read_heads * self.memory_vector_dim)
        # Strengths for read heads
        self.read_strengths = tf.keras.layers.Dense(self.num_read_heads)

        # Key vector for write head
        self.write_key = tf.keras.layers.Dense(self.memory_vector_dim)
        # Strength for write head
        self.write_strength = tf.keras.layers.Dense(1)

        # Erase and add vectors for write operation
        self.erase_vector = tf.keras.layers.Dense(self.memory_vector_dim, activation="sigmoid")
        self.add_vector = tf.keras.layers.Dense(self.memory_vector_dim)
        
        # Final output layer
        self.output_layer = tf.keras.layers.Dense(self.input_dim)

    def forward(self, x, prev_state):
        """
        Defines a single step of the DNC.

        Args:
            x (tf.Tensor): input tensor of shape (batch_size, 12), as per our data flow plan.
            prev_state (DNCState): previous state of the DNC, which includes memory, read vectors, read and write weights, 
                                   the temporal linkage matrix, precedence weights, and the controller state.

        Returns:
            tuple: output and updated state of the DNC. The output is a tensor of shape (batch_size, 464), 
                   as per our data flow plan, and the state is an instance of DNCState, containing the updated state of the DNC.
        """

        # Unpack previous state
        prev_memory, prev_read_vectors, prev_read_weights, prev_write_weights, prev_linkage_matrix, prev_precedence_weights, prev_controller_state = prev_state


        # Pass input through controller. The controller takes in the input and the previous state and outputs a tensor of
        # shape (batch_size, 256) and an updated controller state. This is step 1 of our data flow plan.
        controller_output, hidden_state, cell_state = self.controller(x, prev_controller_state)

        # Calculate read and write weights, and read vectors. This is based on the controller output and the previous state.
        # These weights and vectors will be used in reading from and writing to memory. This corresponds to steps 2 and 3 
        # of our data flow plan. 
        read_weights, write_weights, read_vectors = self._calculate_read_and_write(
                    controller_output, 
                    prev_read_vectors,
                    prev_memory, # Pass prev_memory here
                    prev_read_weights,
                    prev_linkage_matrix
                )

        read_vectors = self.memory.read(prev_memory, read_weights)

        if read_vectors is None:
            raise ValueError("Read vectors cannot be None")

            print("Read vectors shape:", read_vectors.shape)

        # Update memory. The memory is updated based on the previous memory, the write weights, and the controller output. 
        # This corresponds to step 5 of our data flow plan.
        erase_vector = self.erase_vector(controller_output)
        add_vector = self.add_vector(controller_output)

        
        memory = self._update_memory(prev_memory, write_weights, erase_vector, add_vector)

        # Update linkage matrix. The linkage matrix is updated based on the previous and current write weights. 
        # This is part of step 3 of our data flow plan.
        linkage_matrix = self._update_linkage_matrix(prev_write_weights, write_weights, prev_precedence_weights, prev_linkage_matrix)

        # Calculate precedence weights. The precedence weights are updated based on the previous precedence weights and 
        # the current write weights. This is also part of step 3 of our data flow plan.
        precedence_weights = self._calculate_precedence_weights(prev_precedence_weights, write_weights)

        # Prepare output. The output is formed by concatenating the controller output, read vectors, and precedence weights,
        # and passing them through a linear layer. This corresponds to step 4 of our data flow plan.
        output = self._prepare_output(controller_output, read_vectors, precedence_weights)

        # Pack updated state. The updated state includes the updated memory, read vectors, read and write weights, temporal 
        # linkage matrix, precedence weights, and controller state. This state will be passed into the next time step.
        state = DNCState(
            memory=memory,
            read_vectors=read_vectors,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage_matrix=linkage_matrix,
            precedence_weights=precedence_weights,
            controller_state=controller_state,
        )

        return output, state
        
    def _calculate_read_and_write(self, controller_output, prev_read_vectors, memory_matrix, prev_read_weights, prev_linkage_matrix):
        """
        Calculate read and write weights and read vectors for the current time step.

        Args:
            controller_output (tf.Tensor): output of the controller from the current time step, shape (batch_size, 256).
            prev_read_vectors (tf.Tensor): read vectors from the previous time step, shape (batch_size, 4, 20).
            prev_memory (tf.Tensor): memory matrix from the previous time step, shape (batch_size, 128, 20).
            prev_read_weights (tf.Tensor): read weights from the previous time step, shape (batch_size, 4, 128).
            prev_linkage_matrix (tf.Tensor): temporal linkage matrix from the previous time step, shape (batch_size, 128, 128).

        Returns:
            tuple: read_weights, write_weights, and read_vectors for the current time step.
                read_weights (tf.Tensor): shape (batch_size, 4, 128).
                write_weights (tf.Tensor): shape (batch_size, 128).
                read_vectors (tf.Tensor): shape (batch_size, 4, 20).
        """
        print("Memory matrix shape:", memory_matrix.shape)

        # We calculate the interface parameters from the controller output. These are the parameters for the read and 
        # write heads, and they include the key vectors, strengths, and interpolation gate for each head. 
        # This corresponds to step 2 of our data flow plan.
        read_params, write_params = self.controller_to_interface(controller_output)

        # We then calculate the read and write weights based on the previous memory, read weights, the temporal linkage 
        # matrix, and the read and write parameters. The 'calculate_read_weights' and 'calculate_write_weights' methods 
        # may involve operations such as content-based addressing and temporal linking. 
        read_weights = self.calculate_read_weights(read_params, memory_matrix, prev_read_weights, prev_linkage_matrix)
        write_weights = self.calculate_write_weights(write_params, memory_matrix)


        # Once we have the read weights, we can read from memory. We retrieve information from memory based on the read weights,
        # resulting in read vectors. This corresponds to step 3 of our data flow plan.
        # Memory.read()
        print("Memory matrix shape:", memory_matrix.shape)
        print("Read weights shape:", read_weights.shape)

        assert memory_matrix.shape[-1] == self.memory.word_size, "Last dimension must match word size"


        read_vectors = self.memory.read(memory_matrix, read_weights)

        return read_weights, write_weights, read_vectors
    
    def _update_memory(self, prev_memory, write_weights, erase_vector, add_vector):
        """
        Update the memory matrix.

        Args:
            prev_memory (tf.Tensor): memory matrix from the previous time step, shape (batch_size, 128, 20).
            write_weights (tf.Tensor): write weights for the current time step, shape (batch_size, 128).
            erase_vector (tf.Tensor): erase vector for the current time step, shape (batch_size, 20).
            add_vector (tf.Tensor): add vector for the current time step, shape (batch_size, 20).

        Returns:
            tf.Tensor: updated memory matrix, shape (batch_size, 128, 20).
        """

        # We first expand the dimensions of the write weights and the erase vector to match the memory matrix.
        # This is necessary because these tensors are used to update the memory matrix element-wise.
        expanded_write_weights = tf.expand_dims(write_weights, axis=-1)  # shape (batch_size, 128, 1)
        expanded_erase_vector = tf.expand_dims(erase_vector, axis=1)  # shape (batch_size, 1, 20)

        # We then calculate the retention vector, which determines the extent to which the memory at each location is preserved.
        # The retention vector is the complement of the element-wise product of the write weights and the erase vector.
        retention_vector = 1 - expanded_write_weights * expanded_erase_vector  # shape (batch_size, 128, 20)

        # The memory matrix is updated by applying the retention vector and adding the weighted add vector.
        # This corresponds to step 5 of our data flow plan.
        memory = prev_memory * retention_vector + tf.expand_dims(write_weights, axis=-1) * tf.expand_dims(add_vector, axis=1)

        return memory
    
    def _update_linkage_matrix(self, prev_write_weights, write_weights, prev_precedence_weights, prev_linkage_matrix):
        """
        Update the linkage matrix.

        Args:
            prev_write_weights (tf.Tensor): write weights from the previous time step, shape (batch_size, 128).
            write_weights (tf.Tensor): write weights for the current time step, shape (batch_size, 128).

        Returns:
            tf.Tensor: updated linkage matrix, shape (batch_size, 128, 128).
        """
        
        # First, we'll update the precedence weight p using the formula:
        # p_t = (1 - sum_t(w_t)) * p_{t-1} + w_t
        # where w_t is the write weight at time t.
        
        prev_summed_write_weights = tf.reduce_sum(prev_write_weights, axis=1, keepdims=True)
        precedence_weight = (1 - prev_summed_write_weights) * prev_precedence_weights + prev_write_weights

        # Update the linkage matrix L using the formula:
        # L_t(i, j) = (1 - w_t(i) - w_t(j)) * L_{t-1}(i, j) + w_t(i) * p_{t-1}(j)
        # This formula updates the linkage matrix based on the current write weights and the precedence weights.
        
        expanded_write_weights = tf.expand_dims(write_weights, 2)  # Expanding to shape (batch_size, 128, 1)
        expanded_precedence_weight = tf.expand_dims(prev_precedence_weights, 1)
        # Expanding to shape (batch_size, 1, 128)
        
        # Calculating the updated values for the linkage matrix
        print("Shape of expanded_write_weights:", expanded_write_weights.shape)

        L = (1 - expanded_write_weights - tf.transpose(expanded_write_weights, perm=[0, 2, 3, 1])) * prev_linkage_matrix + expanded_write_weights * expanded_precedence_weight


        # Update the precedence weight and linkage matrix for the next time step
        #self.prev_precedence_weight = precedence_weight
        #self.prev_linkage_matrix = L

        return L
    
    def _calculate_precedence_weights(self, prev_precedence_weights, write_weights):
        """
        Calculate the precedence weights.

        Args:
            prev_precedence_weights (tf.Tensor): precedence weights from the previous time step, shape (batch_size, 128).
            write_weights (tf.Tensor): write weights for the current time step, shape (batch_size, 128).

        Returns:
            tf.Tensor: updated precedence weights, shape (batch_size, 128).
        """
        
        # Calculating the sum across the memory locations of the write weights
        summed_write_weights = tf.reduce_sum(write_weights, axis=1, keepdims=True)
        
        # Updating the precedence weights
        precedence_weights = (1 - summed_write_weights) * prev_precedence_weights + write_weights

        return precedence_weights

    def _prepare_output(self, controller_output, read_vectors, precedence_weights):
        """
        Prepare the output tensor.

        Args:
            controller_output (tf.Tensor): Output from the controller, shape (batch_size, 256).
            read_vectors (tf.Tensor): Read vectors from the memory, shape (batch_size, 4, 20).
            precedence_weights (tf.Tensor): Precedence weights, shape (batch_size, 128).

        Returns:
            tf.Tensor: Final DNC output, shape depending on the output_dimension of the linear layer.
        """

        # Reshape read_vectors from (batch_size, 4, 20) to (batch_size, 80)
        read_vectors_reshaped = tf.reshape(read_vectors, [tf.shape(read_vectors)[0], -1])

        # Concatenate along the second dimension (axis=1)
        concatenated_output = tf.concat([controller_output, read_vectors_reshaped, precedence_weights], axis=1)

        # Pass the concatenated tensor through the linear layer
        final_output = self.output_layer(concatenated_output)

        return final_output
    
    def controller_to_interface(self, controller_output):
        """
        Map the controller's output to the interface parameters for the read and write heads.

        Args:
            controller_output (tf.Tensor): output of the controller, shape (batch_size, 256).

        Returns:
            tuple: Parameters for read and write heads.
        """
        # Compute the read keys
        read_keys_out = self.read_keys(controller_output)
        read_keys_out = tf.reshape(read_keys_out, [-1, self.num_read_heads, self.memory_vector_dim])

        # Compute the read strengths
        read_strengths_out = self.read_strengths(controller_output)

        # Combine read parameters
        read_params = read_keys_out, read_strengths_out
        
        # Compute write strength 
        write_strength = self.write_strength(controller_output)

        write_strength = tf.squeeze(write_strength, axis=-1) 

        print("Write strength shape:", write_strength.shape)
        # Compute write parameters as before
        
        write_params = self.write_key(controller_output), self.write_strength(controller_output)
        

        return read_params, write_params

    def calculate_read_weights(self, read_params, prev_memory, prev_read_weights, prev_linkage_matrix):
        """
        Compute read weights for memory based on content-based addressing and temporal links.

        Args:
            read_params (tuple): Parameters for the read head, obtained from controller_to_interface.
            prev_memory (tf.Tensor): Memory matrix from the previous time step, shape (batch_size, 128, 20).
            prev_read_weights (tf.Tensor): Read weights from the previous time step, shape (batch_size, 4, 128).
            prev_linkage_matrix (tf.Tensor): Temporal linkage matrix from the previous time step, shape (batch_size, 128, 128).

        Returns:
            tf.Tensor: Read weights, shape (batch_size, 4, 128).
        """
        # Implement content-based addressing, backward and forward temporal links to compute read weights
        # Note: This is a simplistic implementation. In practice, more sophisticated mechanisms are used.
        read_keys, read_strengths = read_params
        
        
        content_based_weights = self.memory.content_based_addressing(
                            prev_memory,
                            read_keys,
                            read_strengths,
                            self.num_read_heads)


      
        # Using temporal linkage for forward and backward weights can be complex. As a simplification, we're only using content-based addressing here.
        read_weights = content_based_weights

        return read_weights

    def calculate_write_weights(self, write_params, prev_memory):
        """
        Compute write weights for memory based on content-based addressing.

        Args:
            write_params (tuple): Parameters for the write head, obtained from controller_to_interface.
            prev_memory (tf.Tensor): Memory matrix from the previous time step, shape (batch_size, 128, 20).

        Returns:
            tf.Tensor: Write weights, shape (batch_size, 128).
        """
        # Implement content-based addressing to compute write weights
        # Note: This is a simplistic implementation. In practice, more sophisticated mechanisms are used.

        write_key, write_strength = write_params
        write_weights = self.memory.content_based_addressing(
                prev_memory, 
                write_key, 
                write_strength,
                num_read_heads=1)

        return write_weights

    #def zero_state(self, batch_size):
            # Return an initial state of zeros

    #def reset_states(self):
            # Reset states of all components
                    
class Memory(tf.Module):
    def __init__(self, memory_size: int, word_size: int):
        super().__init__()
        self.memory_size = memory_size  # Number of memory slots
        self.word_size = word_size  # Size of each memory slot
        
        # It might be useful to initialize the memory matrix here if needed
        # self.memory_matrix = ...

    def content_based_addressing(self, memory_matrix, keys, strengths, num_read_heads):
        """
        Compute content-based addressing weights.
        
        Args:
            memory_matrix (tf.Tensor): Memory matrix from the previous time step, shape (batch_size, memory_size, word_size).
            keys (tf.Tensor): Keys for addressing, shape can be (batch_size, num_keys, word_size) where num_keys is 1 for writing and can be >1 for reading.
            strengths (tf.Tensor): Strengths (beta) for sharpening, shape can be (batch_size, num_keys).

        Returns:
            tf.Tensor: Addressing weights, shape (batch_size, num_keys, memory_size) for reading or (batch_size, memory_size) for writing.
        """
        print("Memory matrix shape before norm:", memory_matrix.shape) 
        print("Keys shape before norm:", keys.shape)
        print("Strengths shape:", strengths.shape)
        
        strengths = tf.reshape(strengths, [-1, num_read_heads])

        # Normalize 
        memory_norm = tf.nn.l2_normalize(memory_matrix, axis=-1)
        key_norm = tf.nn.l2_normalize(keys, axis=-1)

        print("Memory matrix shape after norm:", memory_norm.shape)
        print("Keys shape after norm:", key_norm.shape)
        
        
        print("key_norm shape:", tf.shape(key_norm))
        print("memory_norm shape:", tf.shape(memory_norm))
        # Compute the similarity using cosine similarity
        similarity = tf.einsum('bik,bjk->bij', key_norm, memory_norm)  # using einsum to calculate dot product
        
        # Sharpen the similarity with strengths
        sharpened_similarity = similarity * strengths[:, :, tf.newaxis]

        # Use softmax to produce the weights
        weights = tf.nn.softmax(sharpened_similarity, axis=-1)
        
        return weights
   
    def read(self, memory_matrix, read_weights):

        read_vectors = []

        for i in range(read_weights.shape[1]):

            # Extract read weights for current head
            head_read_weights = read_weights[:, i, :]

            # Reshape weight matrix for multiplication
            head_read_weights = tf.reshape(head_read_weights, [tf.shape(head_read_weights)[0], -1, 1]) 

            # Transpose weight matrix  
            head_read_weights = tf.transpose(head_read_weights, perm=[0, 2, 1])

            # Matrix multiplication to get read vector
            head_read_vector = tf.matmul(head_read_weights, memory_matrix)
            
            read_vectors.append(head_read_vector)
        
        # Stack the per-head read vectors into one tensor
        read_vectors = tf.stack(read_vectors, axis=1)

        print("Memory matrix shape:", memory_matrix.shape)
        print("Read weights shape:", read_weights.shape)

        assert read_vectors is not None, "Read vectors cannot be None"  

        print("Read vectors shape:", read_vectors.shape)

        return read_vectors