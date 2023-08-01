import os
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from night_two.memory.memory_matrix import MemoryMatrix
from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit
from night_two.memory.temporal_linkage_matrix import TemporalLinkageMemoryUnit
from night_two.memory.dnc_memory import ContentAddressableDNC, ContentAddressableReadHeadWithLinkage, ReadHead, WriteHead, DNC, ContentAddressableWriteHeadWithLinkage, Controller

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

class TestContentAddressableWriteHeadWithLinkage(unittest.TestCase):
    def setUp(self):
        # Set up some parameters for the tests
        self.memory_size = 8
        self.num_memory_slots = 4
        self.batch_size = 2

        # Create an instance of the write head for the tests
        self.write_head = ContentAddressableWriteHeadWithLinkage(self.memory_size, self.num_memory_slots, self.batch_size)

    def test_call(self):
        # Initialize memory with random values
        initial_memory = tf.random.uniform(shape=(self.batch_size, self.num_memory_slots, self.memory_size), minval=-1, maxval=1)

        # Create some fake controller outputs
        controller_output = tf.random.uniform(shape=(self.batch_size, 3*self.memory_size), minval=-1, maxval=1)

        # Call the write head with the initial memory and controller outputs
        new_memory = self.write_head.call(initial_memory, controller_output)

        # Check the shape of the updated memory
        self.assertEqual(new_memory.shape, initial_memory.shape)

        # Check the write weights
        write_weights = self.write_head.get_write_weights()
        self.assertEqual(write_weights.shape, (self.batch_size, self.num_memory_slots))
        self.assertTrue(np.allclose(np.sum(write_weights.numpy(), axis=-1), np.ones(self.batch_size)))

        # Check the temporal linkage matrix
        temporal_linkage_matrix = self.write_head.get_temporal_linkage_matrix()
        self.assertEqual(temporal_linkage_matrix.shape, (self.batch_size, self.num_memory_slots, self.num_memory_slots))

    def test_reset_states(self):
        # Initialize memory with random values
        initial_memory = tf.random.uniform(shape=(self.batch_size, self.num_memory_slots, self.memory_size), minval=-1, maxval=1)

        # Create some fake controller outputs
        controller_output = tf.random.uniform(shape=(self.batch_size, 3*self.memory_size), minval=-1, maxval=1)

        # Call the write head with the initial memory and controller outputs
        _ = self.write_head.call(initial_memory, controller_output)

        # Call the reset_states method
        self.write_head.reset_states()

        # Check that the previous write weights have been reset
        prev_write_weights = self.write_head.get_prev_write_weights()
        self.assertTrue(np.allclose(prev_write_weights.numpy(), np.zeros((self.batch_size, self.num_memory_slots))))

        # Check that the temporal linkage matrix has been reset
        temp_linkage_matrix = self.write_head.get_temporal_linkage_matrix()
        self.assertTrue(np.allclose(temp_linkage_matrix.numpy(), np.zeros((self.batch_size, self.num_memory_slots, self.num_memory_slots))))
             
class TestContentAddressableReadHeadWithLinkage(unittest.TestCase):
    def setUp(self):
        self.memory_size = 5
        self.num_memory_slots = 10
        self.batch_size = 2
        self.controller_output_size = self.memory_size + 1  # Key size + strength

        self.memory = tf.random.normal(shape=(self.batch_size, self.num_memory_slots, self.memory_size))
        self.controller_output = tf.random.normal(shape=(self.batch_size, self.controller_output_size))

        self.read_head = ContentAddressableReadHeadWithLinkage(self.memory_size, self.num_memory_slots, self.batch_size)
        # Perform an initial update to avoid None values
        self.read_head(self.memory, self.controller_output)
        
    def test_call_output_shape(self):
        output = self.read_head(self.memory, self.controller_output)
        self.assertEqual(output.shape, (self.batch_size, self.memory_size))

    def test_temporal_linkage_matrix_update(self):
        initial_matrix = self.read_head.get_temporal_linkage_matrix()
        self.read_head(self.memory, self.controller_output)
        updated_matrix = self.read_head.get_temporal_linkage_matrix()
        self.assertNotEqual(np.sum(initial_matrix - updated_matrix), 0)

    def test_read_weights_update(self):
        self.read_head(self.memory, self.controller_output)
        initial_weights = self.read_head.get_prev_read_weights()

        # Create a new controller output
        self.controller_output = tf.random.normal(shape=(self.batch_size, self.controller_output_size))

        self.read_head(self.memory, self.controller_output)
        updated_weights = self.read_head.get_prev_read_weights()

        self.assertNotEqual(np.sum(initial_weights - updated_weights), 0)

    def test_call_consistency(self):
        output_1 = self.read_head(self.memory, self.controller_output)
        output_2 = self.read_head(self.memory, self.controller_output)
        self.assertTrue(np.allclose(output_1, output_2))

    def test_reset_states(self):
        self.read_head(self.memory, self.controller_output)
        self.read_head.reset_states()
        reset_weights = self.read_head.get_prev_read_weights()
        reset_matrix = self.read_head.get_temporal_linkage_matrix()
        self.assertTrue(np.all(reset_weights == 0))
        self.assertTrue(np.all(reset_matrix == 0))
        
    def test_read_weights_sum(self):
        self.read_head(self.memory, self.controller_output)
        read_weights = self.read_head.get_read_weights()
        self.assertTrue(np.allclose(np.sum(read_weights.numpy(), axis=-1), np.ones(self.batch_size)))
        
    def test_read_vectors_shape(self):
        read_vectors = self.read_head(self.memory, self.controller_output)
        self.assertEqual(read_vectors.shape, (self.batch_size, self.memory_size))    
      
class TestController(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.controller_output_size = 30
        self.read_interface_size = 5
        self.write_interface_size = 5
        self.batch_size = 2

        self.controller = Controller(input_dim=self.input_size, controller_units=self.hidden_size, controller_output_size=self.controller_output_size)

        self.inputs = tf.random.normal(shape=(self.batch_size, self.input_size))
        self.read_vectors = tf.random.normal(shape=(self.batch_size, self.read_interface_size))

    def test_output_shape(self):
        output, (state_h, state_c) = self.controller(self.inputs)
        output = tf.squeeze(output)  # This line removes dimensions of size 1
        expected_output_shape = (self.batch_size, self.controller_output_size)
        self.assertEqual(output.shape, expected_output_shape)

    def test_lstm_states_shape(self):
        _, (state_h, state_c) = self.controller(self.inputs)
        expected_state_shape = (self.batch_size, self.hidden_size)
        self.assertEqual(state_h.shape, expected_state_shape)
        self.assertEqual(state_c.shape, expected_state_shape)

    def test_output_consistency(self):
        output_1, _ = self.controller(self.inputs)
        output_2, _ = self.controller(self.inputs)
        self.assertTrue(np.allclose(output_1.numpy(), output_2.numpy()))   


class TestDNCTwo(unittest.TestCase):
    
    def setUp(self):
        self.memory_size = 10
        self.num_memory_slots = 20
        self.input_size = 5
        self.controller_units = 50
        self.num_read_heads = 3
        self.num_write_heads = 2
        self.batch_size = 4
        self.num_indicators = 2
        self.num_settings_per_indicator = 3

        self.dnc = DNCTwo(self.memory_size, self.num_memory_slots, self.input_size, self.controller_units, 
                          self.num_read_heads, self.num_write_heads, self.batch_size, 
                          self.num_indicators, self.num_settings_per_indicator)

    def test_initialization(self):
        self.assertIsInstance(self.dnc.controller, Controller)
        self.assertEqual(len(self.dnc.read_heads), self.num_read_heads)
        self.assertEqual(len(self.dnc.write_heads), self.num_write_heads)
        self.assertEqual(self.dnc.memory.shape, (self.batch_size, self.num_memory_slots, self.memory_size))

    def test_call(self):
        input_data = tf.random.uniform((self.batch_size, self.input_size))
        output = self.dnc(input_data)

        # Check output shape
        expected_output_dim = self.dnc.controller.fc.units + self.num_read_heads * self.memory_size
        self.assertEqual(output.shape, (self.batch_size, expected_output_dim))

    def test_reset_states(self):
        # Call dnc to modify states
        input_data = tf.random.uniform((self.batch_size, self.input_size))
        self.dnc(input_data)

        self.dnc.reset_states()

        for read_head in self.dnc.read_heads:
            self.assertTrue(tf.reduce_all(read_head.get_read_weights() == tf.zeros((self.batch_size, self.num_memory_slots))))
            self.assertIsNone(read_head.get_temporal_linkage_matrix())

        for write_head in self.dnc.write_heads:
            self.assertTrue(tf.reduce_all(write_head.get_write_weights() == tf.zeros((self.batch_size, self.num_memory_slots))))
            self.assertIsNone(write_head.get_temporal_linkage_matrix())

    def test_reset_memory(self):
        # Call dnc to modify memory
        input_data = tf.random.uniform((self.batch_size, self.input_size))
        self.dnc(input_data)

        self.dnc.reset_memory()

        self.assertTrue(tf.reduce_all(self.dnc.memory == tf.zeros((self.batch_size, self.num_memory_slots, self.memory_size))))


if __name__ == '__main__':
    unittest.main()