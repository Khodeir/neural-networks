from numpy import *
from layer import *
import matplotlib.pyplot as plt


class NeuralNet(object):
    '''A general neural network class.'''
    def __init__(self, layers, weights=None):
        '''Initialize a neural network where layer_sizes describes the number of units in each layer respectively from the bottom-up.'''
        layer_sizes = [layer.size for layer in layers]
        self.numlayers = len(layers)
        self.layers = layers
        if weights is None:
            self.weights = [make_matrix(layer_sizes[i], layer_sizes[i+1]) for i in range(self.numlayers - 1)]
        else:
            self.weights = weights

    def add_layer(self, layer):
        '''Adds a layer to the top of the network and initializes a relevant weight matrix'''
        i = self.numlayers-1  # the index of the old top layer
        self.layers.append(layer)
        self.weights.append(make_matrix(self.layers[i].size, self.layers[i+1].size))
        self.numlayers += 1

    def forward_pass(self, data, skip_layer=0):
        '''The forward pass passes information into the bottom layer of the net and propagates the data up the net'''
        last = len(self.weights)
        if skip_layer > 0:
            data = dot(data, self.weights[skip_layer-1])
        for i in range(skip_layer, self.numlayers):
            data = self.layers[i].process(data)
            if i < last:
                data = dot(data, self.weights[i])
        return [layer.activities for layer in self.layers]

    def backward_pass(self, data, skip_layer=0):
        '''The backward pass pretends that data is the state of the top (output) layer and propagates it down the net'''

        last = len(self.weights)
        if skip_layer > 0:
            data = dot(data, self.weights[last-skip_layer].transpose())
        for i in range(0, self.numlayers - skip_layer)[::-1]:
            data = self.layers[i].process(data)
            if i > 0:
                data = dot(data, self.weights[i-1].transpose())
        return [layer.activities for layer in self.layers]

    def get_layers(self):
        return self.layers

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        '''Be careful using this. It could result in bad bugs, I just need this for gradcheck'''
        self.weights = weights

    def feature_map(self, i, (featN, featM), (mapN, mapM)):
        '''Returns a map of the i'th weight matrix in the network. Each unit in the i+1'th layer corresponds to a
        (featN x featM = layer[i].size) map and there are (mapN x mapM = layer[i+1].size) maps.'''
        assert self.layers[i].size == featN * featM, 'Number of units in the ith layer must equal featN*featM'
        assert self.layers[i+1].size == mapN * mapM, 'Number of units in the i+1th layer must equal mapN*mapM'
        vh = transpose(self.weights[i])
        result = zeros((featN*mapN, featM*mapM))
        row = 0
        col = 0
        for i in range(self.layers[i+1].size):
            result[row:row+featN, col:col+featM] = vh[i].reshape((featN, featM))
            col += featM
            if col == featM*mapM:
                row += featN
                col = 0
        return result


def dropout(data, rate=0.2):
    rate = 2*rate
    drop = (rate*random.random((data.shape)) > random.random((data.shape))).astype(int)
    return data - data * drop


def make_matrix(insize, outsize):
    '''Random initialisation using heuristic. Uniform distribution [-e,e].'''
    epsilon = sqrt(6)/sqrt(insize+outsize)
    weight_matrix = 2*epsilon*random.rand(insize, outsize) - epsilon
    return weight_matrix


def draw(data):
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.show()
