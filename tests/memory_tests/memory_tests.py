import os
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import torch
from night_two.memory.memory_matrix import MemoryMatrix
from night_two.memory.content_addressable_memory import ContentAddressableMemoryUnit
from night_two.memory.temporal_linkage_matrix import TemporalLinkageMemoryUnit
from night_two.memory.dnc_memory import ContentAddressableDNC, ReadHead, WriteHead, DNC

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
            self.memory_unit.write([torch.Tensor([i])])

    def test_capacity(self):
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        self.memory_unit.write([torch.Tensor([5])])
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        assert self.memory_unit.memory[0][0].item() == 1

    def test_write(self):
        with self.assertRaises(TypeError):
            self.memory_unit.write("Not a list")
        with self.assertRaises(TypeError):
            self.memory_unit.write([1, 2, 3])

    def test_read(self):
        episode = self.memory_unit.read(2)
        assert episode[0].item() == 2
        with self.assertRaises(IndexError):
            self.memory_unit.read(-1)
        with self.assertRaises(IndexError):
            self.memory_unit.read(self.memory_unit.capacity)

    def test_predecessor_successor(self):
        assert self.memory_unit.predecessor(0) is None
        assert self.memory_unit.successor(self.memory_unit.capacity - 1) is None
        assert self.memory_unit.predecessor(2)[0].item() == 1
        assert self.memory_unit.successor(2)[0].item() == 3

    def test_save_load(self):
        filename = "test_memory.pt"
        self.memory_unit.save(filename)
        assert os.path.exists(filename)
        new_memory_unit = TemporalLinkageMemoryUnit(self.memory_unit.capacity)
        new_memory_unit.load(filename)
        for i in range(self.memory_unit.capacity):
            assert self.memory_unit.read(i)[0].item() == new_memory_unit.read(i)[0].item()
        os.remove(filename)

class TestContentAddressableMemoryUnit(unittest.TestCase):
    
    def setUp(self):
        self.capacity = 5
        self.memory_unit = ContentAddressableMemoryUnit(self.capacity)
        for i in range(self.capacity):
            self.memory_unit.write([np.array([i])])

    def test_capacity(self):
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        self.memory_unit.write([np.array([5])])
        assert len(self.memory_unit.memory) == self.memory_unit.capacity
        assert np.array_equal(self.memory_unit.memory[0][0], np.array([1]))

    def test_write_read(self):
        episode = [np.array([6])]
        self.memory_unit.write(episode)
        read_episode = self.memory_unit.read(self.capacity - 1)
        assert np.array_equal(read_episode[0], episode[0])

    def test_duplicate_content(self):
        episode = [np.array([6])]
        with self.assertRaises(ValueError):
            self.memory_unit.write(episode)
            self.memory_unit.write(episode)  # Duplicate write should raise ValueError

    def test_retrieve(self):
        episode = [np.array([7])]
        self.memory_unit.write(episode)
        retrieved_episode = self.memory_unit.retrieve([np.array([7])])  # Pass np.array([7]) instead of [7]
        assert np.array_equal(retrieved_episode[0], episode[0])
        with self.assertRaises(KeyError):
            self.memory_unit.retrieve([np.array([100])])  # Pass np.array([100]) instead of [100]

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
        self.assertEqual(self.model.memory.shape, (100, 20))
        
    def test_read_heads(self):
        self.assertEqual(len(self.model.read_heads), 2)

    def test_write_heads(self):
        self.assertEqual(len(self.model.write_heads), 2)

    def test_model_output_shape(self):
        input_data = tf.random.normal((1, 10, 128))  # batch_size, sequence_length, input_dim
        output_data = self.model(input_data)
        self.assertEqual(output_data.shape, (1, 10, 128))  # batch_size, sequence_length, output_dim

if __name__ == '__main__':
    unittest.main()