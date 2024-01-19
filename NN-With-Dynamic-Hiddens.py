# Import necessary libraries
import numpy as np
from scipy.special import expit

class NeuralNetwork:
    # Setting up a neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.HL = [] # In loop list for hidden layers
        self.HLCount = 0 # In loop layer count(number)
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.activation_function = lambda x : expit(x)
        # Input to Hidden Weights
        self.w_i_h = np.random.default_rng().normal(0, pow(self.input_nodes, -0.5),(self.hidden_nodes[0], self.input_nodes))
        # Hidden to Hidden Weights
        if len(self.hidden_nodes) > 1: # If there is mare than one hidden layer
            self.PrevHL = self.hidden_nodes[0]
            del self.hidden_nodes[0]
            for i in self.hidden_nodes:
                self.HL.append(np.random.default_rng().normal(0, pow(self.PrevHL, -0.5),(i, self.PrevHL)))
                self.HLCount += 1
                self.PrevHL = i
            # Hidden to Output Weights
            self.w_h_o = np.random.default_rng().normal(0, pow(self.PrevHL, -0.5),(self.output_nodes, self.PrevHL))
        elif len(self.hidden_nodes) == 1: # If there is one hidden layer
            # Hidden to Output Weights
            self.w_h_o = np.random.default_rng().normal(0, pow(self.hidden_nodes[0], -0.5),(self.output_nodes, self.hidden_nodes[0]))
        pass
       
    def query(self, input_list):
        self.OL = [] # In loop output list
        self.OCount = 0 # In loop output count
        # Convert the list of input values to a two-dimensional array
        inputs = np.array(input_list, ndmin=2).T
        # Calculation of the input signal and then the output of the hidden layer
        # Using dot multiplication
        x_hidden = np.dot(self.w_i_h, inputs)
        o_hidden = self.activation_function(x_hidden)
        # Calculation of the input signal(hidden) and then the output of the hidden layer
        for i in self.HL:
            x_hidden = np.dot(self.HL[self.OCount], o_hidden)
            self.OL.append(self.activation_function(x_hidden))
            o_hidden = self.activation_function(x_hidden)
            self.OCount += 1
        # Calculation of the input signal(hidden) and then the output of the output layer
        x_output = np.dot(self.w_h_o, o_hidden)
        o_output = self.activation_function(x_output)

        return o_output
