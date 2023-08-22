import os
import unittest
import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from night_two.memory.memory_matrix import MemoryMatrix
from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit
from night_two.memory.temporal_linkage_matrix import TemporalLinkageMatrix, TemporalLinkageMemoryUnit
from night_two.memory.dnc_memory import ContentAddressableDNC, DNCState, Memory, ReadHead, WriteHead, DNC, Controller, DNCTwo

class TestMemoryMatrix(unittest.TestCase):

    def test_initialization(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        assert memory.matrix.shape == (N, W)
        assert (memory.matrix == np.zeros((N, W))).all()
        assert memory.current_address == 0
        assert not memory.filled_once
        
    def test_write_and_read(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        data = np.ones(W)
        memory.write(data)
        assert (memory.read(0) == data).all()
        
    def test_write_beyond_size(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        data1 = np.ones(W)
        data2 = np.ones(W) * 2
        for _ in range(N):
            memory.write(data1)
        memory.write(data2)
        assert (memory.read(0) == data2).all()  # the first cell should have been overwritten

    def test_has_been_filled(self):
        N = 10
        W = 5
        memory = MemoryMatrix(N, W)
        assert not memory.has_been_filled()
        data = np.ones(W)
        for _ in range(N):
            memory.write(data)
        assert memory.has_been_filled()
        # add more test methods here

class TestTemporalLinkageMemoryUnit(unittest.TestCase):

    def setUp(self):
        self.capacity = 5
        self.memory_unit = TemporalLinkageMemoryUnit(self.capacity)
        for i in range(self.capacity):
            self.memory_unit.write([tf.convert_to_tensor([i])])

    def test_capacity(self):
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        self.memory_unit.write([tf.convert_to_tensor([5])])
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        assert int(self.memory_unit.memory[0][0].numpy()) == 1

    def test_write(self):
        with self.assertRaises(TypeError):
            self.memory_unit.write("Not a list")
        with self.assertRaises(TypeError):
            self.memory_unit.write([1, 2, 3])

    def test_read(self):
        episode = self.memory_unit.read(2)
        assert int(episode[0].numpy()) == 2
        with self.assertRaises(IndexError):
            self.memory_unit.read(-1)
        with self.assertRaises(IndexError):
            self.memory_unit.read(self.memory_unit.capacity)

    def test_predecessor_successor(self):
        assert self.memory_unit.predecessor(0) is None
        assert self.memory_unit.successor(self.memory_unit.capacity - 1) is None
        assert int(self.memory_unit.predecessor(2)[0].numpy()) == 1
        assert int(self.memory_unit.successor(2)[0].numpy()) == 3

class TestContentAddressableMemoryUnit(unittest.TestCase):
    
    def setUp(self):
        self.capacity = 5
        self.memory_unit = ContentAddressableMemoryUnit(self.capacity)
        for i in range(self.capacity):
            self.memory_unit.write([tf.convert_to_tensor([i])])

    def test_capacity(self):
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        self.memory_unit.write([tf.convert_to_tensor([5])])
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        assert tf.reduce_all(tf.equal(self.memory_unit.memory[0][0], tf.convert_to_tensor([1])))

    def test_write_read(self):
        episode = [tf.convert_to_tensor([6])]
        self.memory_unit.write(episode)
        read_episode = self.memory_unit.read(self.capacity - 1)
        assert tf.reduce_all(tf.equal(read_episode[0], episode[0]))

    def test_duplicate_content(self):
        episode = [tf.convert_to_tensor([6])]
        with self.assertRaises(ValueError):
            self.memory_unit.write(episode)
            self.memory_unit.write(episode)  # Duplicate write should raise ValueError

    def test_retrieve(self):
        episode = [tf.convert_to_tensor([7])]
        self.memory_unit.write(episode)
        retrieved_episode = self.memory_unit.retrieve([tf.convert_to_tensor([7])])  # Pass tf.convert_to_tensor([7]) instead of [7]
        assert tf.reduce_all(tf.equal(retrieved_episode[0], episode[0]))
        with self.assertRaises(KeyError):
            self.memory_unit.retrieve([tf.convert_to_tensor([100])])  # Pass tf.convert_to_tensor([100]) instead of [100]
                       
class TestDNCModel(unittest.TestCase):
    def setUp(self):
        # Initialize a DNC model
        self.dnc = DNC(10, 5, 1, 1)

    def test_read_head(self):
        # Initialize a read head with a memory size of 5
        read_head = ReadHead(5)

        # Create a mock memory matrix
        memory = tf.random.normal((5, 5))

        # Create a mock controller output
        controller_output = tf.random.normal((1, 10))

        # Perform a read operation
        read_vector = read_head(memory, controller_output)

        # Check that the read vector has the correct shape
        self.assertEqual(read_vector.shape, (5,))

    def test_write_head(self):
        # Initialize a write head with a memory size of 5
        write_head = WriteHead(5)

        # Create a mock memory matrix
        memory = tf.random.normal((5, 5))

        # Create a mock controller output
        controller_output = tf.random.normal((1, 10))

        # Perform a write operation
        new_memory = write_head(memory, controller_output)

        # Check that the new memory matrix has the correct shape
        self.assertEqual(new_memory.shape, (5, 5))

    def test_dnc(self):
        # Create a mock input tensor
        inputs = tf.random.normal((1, 10))

        # Add a time dimension to the inputs
        inputs = tf.expand_dims(inputs, 1)

        # Perform a step of DNC
        read_vectors = self.dnc(inputs)

        # Check that the correct number of read vectors is returned
        self.assertEqual(len(read_vectors), self.dnc.num_read_heads)

        # Check that each read vector has the correct shape
        for read_vector in read_vectors:
            self.assertEqual(read_vector.shape, (1, 5))

class TestContentAddressableDNC(unittest.TestCase):
    
    def setUp(self):
        self.model = ContentAddressableDNC(controller_size=128, memory_size=20, num_read_heads=2, num_write_heads=2, capacity=100)

    def test_controller(self):
        self.assertIsInstance(self.model.controller, LSTM)

    def test_memory_initialization(self):
        self.assertIsInstance(self.model.memory, tf.Variable)
        
    def test_read_heads(self):
        self.assertEqual(len(self.model.read_heads), 2)

    def test_write_heads(self):
        self.assertEqual(len(self.model.write_heads), 2)

    def test_model_output_shape(self):
        input_data = tf.random.normal((1, 10, 128))  # batch_size, sequence_length, input_dim
        output_data = self.model(input_data)
        self.assertEqual(output_data.shape, (1, 10, 168))  # batch_size, sequence_length, output_dim
    
    def test_read_head(self):
        read_head = self.model.read_heads[0]
        dummy_memory = tf.random.normal((self.model.num_memory_slots, self.model.memory_size))
        dummy_controller_output = tf.random.normal((1, 10, self.model.controller_size))
        output = read_head(dummy_memory, dummy_controller_output)
        self.assertEqual(output.shape, (1, 10, self.model.memory_size))
        
    def test_write_head(self):
        write_head = self.model.write_heads[0]
        dummy_memory = tf.random.normal((self.model.num_memory_slots, self.model.memory_size))
        dummy_controller_output = tf.random.normal((1, 10, self.model.controller_size))
        output = write_head(dummy_memory, dummy_controller_output)
        self.assertEqual(output.shape, (self.model.num_memory_slots, self.model.memory_size))
        
    def test_content_addressable_memory(self):
        dummy_controller_output = tf.random.normal((1, 10, self.model.controller_size)).numpy().tolist()
        self.model.content_addressable_memory.write(dummy_controller_output)
        self.assertEqual(len(self.model.content_addressable_memory.memory), min(len(dummy_controller_output), self.model.content_addressable_memory.capacity))

class TestController(unittest.TestCase):

    def setUp(self):
        # Set up a simple controller for testing
        self.controller = LSTM(units=256, stateful=False)

    def test_controller_output_shape(self):
        # Prepare a batch of inputs
        input_data = tf.random.uniform((10, 12))  # batch_size is 10
        input_data = tf.expand_dims(input_data, 1)  # Add a time dimension

        # Compute the output
        output = self.controller(input_data)

        # Check the output shape
        self.assertEqual(output.shape, (10, 256))  # batch_size, hidden_state_size

class TestControllerTwo(unittest.TestCase):
    
    def setUp(self):
        # This method is called before each test. You can initialize common setups here.
        self.hidden_size = 256
        self.controller = Controller(self.hidden_size)
        
    def test_controller_initialization(self):
        self.assertEqual(self.controller.lstm.units, self.hidden_size, "Incorrect hidden size")
        
    def test_controller_call(self):
        # Create input tensor
        batch_size = 32 
        input_dim = 12
        inputs = tf.random.normal((batch_size, 1, input_dim))

        # Get initial states
        states = self.controller.get_initial_state(inputs, batch_size)
        
        # Call controller
        outputs, states = self.controller(inputs, states)

        hidden_state, cell_state = states
        
        new_states = (hidden_state, cell_state)

        self.assertEqual(len(new_states), 2, "Expected two states (h and c for LSTM)")

class TestDNCTwo(unittest.TestCase):

    def setUp(self):
        # This method is called before each test. You can initialize common setups here.
        self.input_dim = 12
        self.num_memory_slots = 128
        self.memory_vector_dim = 20
        self.num_read_heads = 4
        self.controller_hidden_size = 256
        self.dnc = DNCTwo(self.input_dim, self.num_memory_slots, self.memory_vector_dim, self.num_read_heads, self.controller_hidden_size)

    def test_dnc_initialization(self):
        # Check individual components
        self.assertEqual(self.dnc.controller.lstm.units, self.input_dim + self.num_read_heads * self.memory_vector_dim, "Incorrect controller hidden size")
        self.assertIsInstance(self.dnc.memory, Memory, "Memory component missing or not initialized")
        self.assertIsInstance(self.dnc.temporal_linkage_matrix, TemporalLinkageMatrix, "TemporalLinkageMatrix component missing or not initialized")
        # ... and so on for other components

    def test_dnc_forward(self):
        batch_size = 32
        inputs = tf.random.normal((batch_size, 1, self.input_dim))

        # Create an initial state
        memory_init = tf.zeros((batch_size, self.num_memory_slots, self.memory_vector_dim))
        read_vectors_init = tf.zeros((batch_size, self.num_read_heads, self.memory_vector_dim))
        read_weights_init = tf.zeros((batch_size, self.num_read_heads, self.num_memory_slots))
        write_weights_init = tf.zeros((batch_size, self.num_memory_slots))
        linkage_matrix_init = tf.zeros((batch_size, self.num_memory_slots, self.num_memory_slots))
        precedence_weights_init = tf.zeros((batch_size, self.num_memory_slots))
        controller_state_init = (tf.zeros((batch_size, self.controller_hidden_size)), tf.zeros((batch_size, self.controller_hidden_size)))  # Assuming it's an LSTM with hidden and cell state

        prev_state = DNCState(
            memory=memory_init,
            read_vectors=read_vectors_init,
            read_weights=read_weights_init,
            write_weights=write_weights_init,
            linkage_matrix=linkage_matrix_init,
            precedence_weights=precedence_weights_init,
            controller_state=controller_state_init
        )

        output, state = self.dnc.forward(inputs, prev_state)
        
        # Assertions
        self.assertEqual(output.shape, (batch_size, 464), "Incorrect output shape")

if __name__ == '__main__':
    unittest.main()