from numpy import *
from network import *


def backprop(network, data, targets):
    '''data is a matrix of nxv training data. targets has all the target values of the data.
    Backprop returns a list of matrices, network_dE_dW.
    The list contains l-matrices, where l is the number of layers in the network.
    Every matrix in the list contains the error derivatives w.r.t every weight connection into a layer, dE/dw,
    normalised by the number of training examples'''
    network_dE_dW, network_dE_dB = [], []

    #Some info about what the network's like
    layers = network.get_layers()
    num_layers = network.numlayers
    weights = network.get_weights()
    layer_activities = network.forward_pass(data)  # List of matrices of unit activities. Contains l-matrices, where l= numLayers. Order is bottom-up.
    # Every matrix is m x n, where m= numTrainingExamples, n= num units in that layer

    # dE/dy for output layer
    # Using cross-entropy error, dE/dy = -t/y + (1-t)/(1-y) - we should consider passing the error function as a parameter
    dE_dY = - targets/layer_activities[num_layers - 1] + (1 - targets)/(1 - layer_activities[num_layers - 1])

    for j in range(num_layers-1, 0, -1):
        dY_dZ = layers[j].gradient()  # get dy/dz from the layer's gradient method
        dE_dZ = dY_dZ * dE_dY

        dE_dY = dot(dE_dZ, weights[j-1].transpose())  # This will be dE/dY used in next layer
        dE_dW = dot(layer_activities[j-1].transpose(), dE_dZ)/data.shape[0]  # Normalised dE/dW matrix for this layer over all training examples
        dE_dB = (dE_dZ.sum(0).reshape((1, layers[j].size)))/data.shape[0]

        network_dE_dW.insert(0, dE_dW)  # Put this at the front of the list of network deltas
        network_dE_dB.insert(0, dE_dB)
    return network_dE_dW, network_dE_dB


def train(net, X, T, learning_rate=0.1):
    '''Perform one iteration of backpropagation training on net using inputs X and targets T and a learning_rate'''
    weight_derivatives, bias_derivatives = backprop(net, X, T)

    for i in range(len(net.weights)):
        assert net.weights[i].shape == weight_derivatives[i].shape, "Something went wrong here. W and dW are mismatched"
        assert net.layers[i+1].bias.shape == bias_derivatives[i].shape, "Something went wrong here. B and dB are mismatched"
        net.weights[i] -= learning_rate * weight_derivatives[i]
        net.layers[i+1].bias -= learning_rate * bias_derivatives[i]
