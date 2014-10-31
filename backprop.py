from numpy import *
from network import *


def backprop(network, data, targets, skip_layers=0):
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

    down_to = num_layers - (num_layers - skip_layers)

    # dE/dy for output layer
    # Using cross-entropy error, dE/dy = -t/y + (1-t)/(1-y) - we should consider passing the error function as a parameter
    dE_dY = - targets/layer_activities[num_layers - 1] + (1 - targets)/(1 - layer_activities[num_layers - 1])

    for j in range(num_layers-1, down_to, -1):
        dY_dZ = layers[j].gradient()  # get dy/dz from the layer's gradient method
        dE_dZ = dY_dZ * dE_dY

        dE_dY = dot(dE_dZ, weights[j-1].transpose())  # This will be dE/dY used in next layer
        dE_dW = dot(layer_activities[j-1].transpose(), dE_dZ)/data.shape[0]  # Normalised dE/dW matrix for this layer over all training examples
        dE_dB = (dE_dZ.sum(0).reshape((1, layers[j].size)))/data.shape[0]

        network_dE_dW.insert(0, dE_dW)  # Put this at the front of the list of network deltas
        network_dE_dB.insert(0, dE_dB)
    return network_dE_dW, network_dE_dB


def train(net, X, T, learning_rate=0.1, decay_rate=0):
    '''Perform one iteration of backpropagation training on net using inputs X and targets T and a learning_rate'''
    weight_derivatives, bias_derivatives = backprop(net, X, T)

    for i in range(len(net.weights)):
        assert net.weights[i].shape == weight_derivatives[i].shape, "Something went wrong here. W and dW are mismatched"
        assert net.layers[i+1].bias.shape == bias_derivatives[i].shape, "Something went wrong here. B and dB are mismatched"

        reg_term = (decay_rate/X.shape[0])*net.weights[i]  # Regularisation term to be added to dE/dW for the given layer

        net.weights[i] -= learning_rate * (weight_derivatives[i] + reg_term)
        net.layers[i+1].bias -= learning_rate * bias_derivatives[i]


def gradcheck(network, layer, data, targets):
    '''Same idea as backprop, just to confirm the gradients'''
    dE_dW = []
    epsilon = 0.0001

    weights = network.get_weights()
    temp_weights = weights[:]  # Make a copy and work with that so nothing weird happens
    wmatrix = temp_weights[layer]  # Look at derivatives for this layer in particular

    for i in range(shape(wmatrix)[0]):
        for j in range(shape(wmatrix)[1]):
            wmatrix[i][j] += epsilon  # w + epsilon
            temp_weights[layer] = wmatrix
            network.set_weights(temp_weights)  # make these the weights for the network
            error1 = error(network, data, targets)

            wmatrix[i][j] -= 2*epsilon  # w - epsilon
            temp_weights[layer] = wmatrix
            network.set_weights(temp_weights)
            error2 = error(network, data, targets)

            grad = ((error1 - error2)/(2*epsilon))/data.shape[0]
            dE_dW.append(grad)

            wmatrix[i][j] += epsilon
            temp_weights[layer] = wmatrix
            network.set_weights(temp_weights)

    network.set_weights(weights)  # Back to original weights
    return dE_dW


def error(net, data, T):
    '''Cross-entropy error. Need this for gradcheck'''
    Y = net.forward_pass(data)[-1]
    error = -(T*log(Y) + (1-T)*log(1-Y)).sum()  # Unregularised error term
    return error


def testNet():
    '''Small multi-layer test net for gradient checking, presumably if this works then any sized net works'''
    net = NeuralNet([LinearLayer(4), LogisticLayer(5), LogisticLayer(3)])
    data = array([[1, 4, 3, 5], [2.2, 3.1, 0.5, -2], [-0.3, 1, 1, 2.1]])  # Data with 3 training examples
    targets = array([[0.2, 0.4, -0.1], [1, 2, 3], [-1, -2, -3]])  # 3 targets

    dE_dW1, dE_dB1 = backprop(net, data, targets)
    print 'Gradients from backprop: '
    print dE_dW1
    print

    for i in range(2):
        print 'Gradients from gradcheck for layer ' + str(i) + ':'
        print gradcheck(net, i, data, targets)
