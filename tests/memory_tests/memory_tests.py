import os
import unittest
from numpy import shape
import tensorflow as tf
from night_two.memory.dnc_memory import DNCState, DNCTwo, Memory, Controller
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes

class TestControllerTwo(unittest.TestCase):

    def test_controller_initialization(self):
        
        controller = Controller(256) 
    
        # Validate hidden size matches plan
        self.assertEqual(controller.hidden_size, 256)

    def test_controller_call(self):

        # Initialize controller
        controller = Controller(hidden_size=256)  

        # Create input  
        input_data = tf.random.uniform((10, 12))
        input_data = tf.expand_dims(input_data, 1)

        # Create dummy states
        states = None

        # Get controller output  
        outputs = controller(input_data, states)[0]


        # Assertions
        self.assertEqual(outputs.shape, (10, 1, 256))

class TestMemory(unittest.TestCase):

  def test_memory_initialization(self):
    # Initialize memory instance to test class attributes
    memory = Memory(128, 20)  

    # Validate memory configuration 
    # Check memory_size and word_size match init
    self.assertEqual(memory.memory_size, 128)
    self.assertEqual(memory.word_size, 20)

  def test_read_write(self):
    
    # Initialize memory for read/write testing
    memory = Memory(128, 20)  

    # Generate dummy memory matrix as input for read
    memory_matrix = tf.random.uniform((10, 128, 20)) 

    # Generate dummy data to write 
    write_data = tf.random.uniform((10, 128, 20))

    # Generate dummy read weights  
    read_weights = tf.random.uniform((10, 4, 128))

    # Write dummy data
    memory.write(write_data)

    # Read using dummy weights 
    read_vectors = memory.read(memory_matrix, read_weights)

    # Validate shape of read vectors
    # This tests the read/write functionality
    # And corresponds to steps 2,3 of data flow plan
    self.assertEqual(read_vectors.shape, (10, 4, 1, 20))

class TestDNCTwo(tf.test.TestCase):
  
  def setUp(self):
    self.input_dim = 12
    self.num_slots = 128
    self.memory_vector_dim = 20
    self.word_size = 20
    self.num_read_heads = 4
    self.controller_hidden_size = 256
    
    self.batch_size = 32
    self.seq_len = 10

    # Create DNC 
    self.dnc = DNCTwo(self.input_dim, self.num_slots, self.memory_vector_dim,
                      self.num_read_heads, self.controller_hidden_size)


  def setup_dnc_model(self):
    # Initialize DNC model 
    self.dnc = DNCTwo(self.input_dim, self.num_memory_slots, ...)

    # Get controller states
    self.initial_controller_state = self.dnc.controller.get_initial_state()  

  def generate_inputs(self):
    # Create input sequence
    inputs = tf.random.uniform((self.batch_size, self.seq_len, self.input_dim))
    return inputs


  def test_forward_pass(self):
    
    # Generate test inputs
    inputs = self.generate_inputs()

    # Initialize previous state
    prev_state = self.dnc.zero_state(self.batch_size, self.num_slots, self.word_size, self.num_read_heads, inputs=inputs)

    # Run forward pass
    outputs, next_state = self.dnc(inputs, prev_state)
    
    # Assert output shape
    expected_output_shape = (self.batch_size, self.seq_len, self.output_dim)
    self.assertEqual(outputs.shape, expected_output_shape)

    # Assert state contents shapes
    expected_memory_shape = (self.batch_size, self.num_slots, self.word_size) 
    self.assertEqual(next_state.memory.shape, expected_memory_shape)
    
    # Assert output shape
    expected_output_shape = (self.batch_size, self.seq_len, self.output_dim)
    self.assertEqual(outputs.shape, expected_output_shape)

    # Assert memory shape 
    expected_memory_shape = (self.batch_size, self.num_slots, self.word_size)
    self.assertEqual(next_state.memory.shape, expected_memory_shape)  

    # Assert read weights shape
    expected_read_weights_shape = (self.batch_size, self.num_read_heads, self.num_slots)
    self.assertEqual(next_state.read_weights.shape, expected_read_weights_shape)

    # Assert controller output shape
    expected_controller_shape = (self.batch_size, self.seq_len, self.controller_hidden_size)
    self.assertEqual(controller_outputs.shape, expected_controller_shape)
      
    # Assert read params shape
    expected_read_params_shape = (self.batch_size, self.num_read_heads, self.word_size)
    self.assertEqual(read_params[0].shape, expected_read_params_shape)

    # Assert concatenated output size
    expected_concat_size = controller_hidden_size + self.num_read_heads*self.word_size + self.num_slots
    self.assertEqual(outputs.shape[-1], expected_concat_size)
    
    
if __name__ == '__main__':
    unittest.main()