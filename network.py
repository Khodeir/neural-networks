from numpy import *
from layer import *
from scipy.io import savemat, loadmat
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
        return data_map(vh, (featN, featM), (mapN, mapM))

    def flatten_parameters(self):
        '''Returns all the network parameters in one vector, weights first, then biases.'''
        parameters = []
        for matrix in self.weights:
            parameters = concatenate((parameters, matrix.flatten()),1)
        for layer in self.layers[1:]:
            parameters = concatenate((parameters, layer.bias.flatten()),1)

        return parameters

    def set_parameters(self, parameters):
        '''Takes flattened network parameters and sets the network's parameters to those.'''
        layer_sizes = [layer.size for layer in self.layers]
        net_size = 0
        for i in range(0, len(layer_sizes)-1):
            net_size += layer_sizes[i]*layer_sizes[i+1]
            net_size += layer_sizes[i+1]

        assert len(parameters) == net_size, "The parameters don't match the dimensions of the net"
        weights = []
        count = 0
        for i in range(0, len(layer_sizes)-1):
            weight = parameters[count: (count + layer_sizes[i]*layer_sizes[i+1])]
            matrix = weight.reshape((layer_sizes[i],layer_sizes[i+1]))
            weights.append(matrix)
            count += (layer_sizes[i]*layer_sizes[i+1])

        for i in range(1,len(layer_sizes)): #Set biases of all the layers, excluding bottom layer
            self.layers[i].bias = (parameters[count:count+layer_sizes[i]])
            count += layer_sizes[i]

        self.weights = weights

    def save_network(self, filename):
        '''Save the network's parameters in a .mat file.'''
        data = {}
        data['numlayers'] = self.numlayers

        for i in range(1, self.numlayers):
            data['weights_' + str(i-1)] = self.weights[i-1]
            data['bias_' + str(i)] = self.layers[i].bias

        savemat(filename, data)

    def load_network(self, filename):
        '''Load network parameters from a .mat file and set it to the parameters of this network.'''
        data = loadmat(filename)
        weights, biases = [],[]
        for i in range(1, self.numlayers):
            self.weights[i-1] = data.get('weights_' + str(i-1))
            self.layers[i].bias = data.get('bias_' + str(i))

    def weight_histogram(self, index=None):
        '''Plot a weight histogram for this net. Optional parameter index to specify a particular weight matrix.'''
        vishid = []
        if index == None:
            for matrix in self.weights[:]:
                flat = matrix.flatten()
                for weight in flat:
                    vishid.append(weight)
        else:
            weights = self.weights[index][:].flatten()
            for weight in weights:
                vishid.append(weight)

        plt.figure()
        plt.xlabel("Size of Weights")
        plt.ylabel("Frequency")
        plt.hist(vishid, bins=100)
        plt.show()
        return vishid

    def bias_histogram(self, index=None):
        '''Plot a bias histogram for this net. Optional parameter index to specify a particular layer.'''
        biases = []
        if index == None:
            for layer in self.layers[:]:
                for bias in layer.bias:
                    biases.append(bias)
        else:
            layer_bias = self.layers[index][:].bias
            for bias in self.layers[index][:].bias:
                biases.append(bias)

        plt.figure()
        plt.xlabel("Size of Weights")
        plt.ylabel("Frequency")
        plt.hist(biases, bins=100)
        plt.show()
        return biases

def dropout(data, rate=0.2):
    drop = random.binomial(1, rate, data.shape)
    return data - data * drop


def make_matrix(insize, outsize):
    '''Random initialisation using heuristic. Uniform distribution [-e,e].'''
    epsilon = sqrt(6)/sqrt(insize+outsize)
    weight_matrix = 2*epsilon*random.rand(insize, outsize) - epsilon
    return weight_matrix

# We should probably move the below functions to another class?

def draw(data):
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.show()


def data_map(data, (featN, featM), (mapN, mapM)):
    result = zeros((featN*mapN, featM*mapM))
    numMaps = mapN * mapM
    row = 0
    col = 0
    for i in range(numMaps):
        result[row:row+featN, col:col+featM] = data[i].reshape((featN, featM))
        col += featM
        if col == featM*mapM:
            row += featN
            col = 0
    return result
