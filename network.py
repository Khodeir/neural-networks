from numpy import *
from layer import *
import matplotlib.pyplot as plt

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
        '''The forward pass passes information into the bottom layer of the net and propagates the data up the net'''
        last = len(self.weights)
        for i in range(self.numlayers):
            data = self.layers[i].process(data)
            if i < last:
                data = dot(data, self.weights[i])
        return [layer.activities for layer in self.layers]

    def backward_pass(self, data):
        '''The backward pass pretends that data is the state of the top (output) layer and propagates it down the net'''
        assert False, "Unimplemented"
        last = len(self.weights)
        for i in range(last-1,-1,-1):
            data = dot(data, self.weights[i].transpose())
            data = self.layers[i].process(data)
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

def draw(data):
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.show()
