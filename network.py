from numpy import *
from layer import *

class NeuralNet(object):
    '''A general neural network class.'''
    def __init__(self, layers):
        '''Initialize a neural network where layer_sizes describes the number of units in each layer respectively from the bottom-up.'''
        layer_sizes = [layer.size for layer in layers]
        self.numlayers = len(layers)
        self.layers = layers
        self.weights = [0.03*random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.numlayers-1)]

    def forward_pass(self, data):
        last = len(self.weights)
        activities = []
        for i in range(self.numlayers):
            data = self.layers[i].process(data)
            activities.append(data)
            if i < last:
                data = dot(data, self.weights[i])
        return activities

    def get_layers(self):
        return self.layers

    def get_weights(self):
        return self.weights
