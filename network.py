from numpy import *
from layer import *


class NeuralNet(object):
    '''A general neural network class.'''
    def __init__(self, layers):
        '''Initialize a neural network where layer_sizes describes the number of units in each layer respectively from the bottom-up.'''
        layer_sizes = [layer.size for layer in layers]
        self.numlayers = len(layers)
        self.layers = layers
        self.weights = []
        for i in range(self.numlayers-1): #Random initialisation using heuristic. Uniform distribution [-e,e].
            epsilon = sqrt(6)/sqrt(layer_sizes[i]+layer_sizes[i+1])
            self.weights.append(2*epsilon*random.rand(layer_sizes[i], layer_sizes[i+1]) - epsilon)

    def forward_pass(self, data):
        last = len(self.weights)
        for i in range(self.numlayers):
            data = self.layers[i].process(data)
            if i < last:
                data = dot(data, self.weights[i])
        return [layer.activities for layer in self.layers]

    def get_layers(self):
        return self.layers

    def get_weights(self):
        return self.weights
