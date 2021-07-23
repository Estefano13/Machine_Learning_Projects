import numpy as np
from neural_nets import *
import unittest

class TestNeuralNetwork(unittest.TestCase):

    def test_relu(self):
        self.assertEqual(rectified_linear_unit(-3), 0)
        self.assertEqual(rectified_linear_unit(0), 0)
        self.assertEqual(rectified_linear_unit(9), 9)

    def test_relu_derivative(self):
        self.assertEqual(rectified_linear_unit_derivative(-3), 0)
        self.assertEqual(rectified_linear_unit_derivative(0), 0)
        self.assertEqual(rectified_linear_unit_derivative(9), 1)

    def test_neural_nets(self):
        expected_input_to_hidden_weights = np.matrix([[1.04754206, 1.06143111], [1.04754206, 1.06143111], [1.04754206, 1.06143111]])
        expected_hidden_to_output_weights = np.matrix([[1.10707298, 1.10707298, 1.10707298]])
        expected_biases = np.matrix([[0.01026478], [0.01026478], [0.01026478]])

        nn = NeuralNetwork()
        nn.train_neural_network()

        self.assertEqual(np.isclose(nn.input_to_hidden_weights, expected_input_to_hidden_weights).all(), True)
        self.assertEqual(np.isclose(nn.hidden_to_output_weights, expected_hidden_to_output_weights).all(), True)
        self.assertEqual(np.isclose(nn.biases, expected_biases).all(), True)

if __name__ == '__main__':
    unittest.main()
