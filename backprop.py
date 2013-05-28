from numpy import *
from network import*

def backprop(network, data, targets):
    '''data is a matrix of nxv training data. targets has all the target values of the data.
    Backprop returns a list of matrices, network_dE_dW.
    The list contains l-matrices, where l is the number of layers in the network.
    Every matrix in the list contains the error derivatives w.r.t every weight connection into a layer, dE/dw,
    normalised by the number of training examples'''
    network_dE_dW = []

    #Some info about what the network's like
    layers = network.get_layers()
    num_layers = len(layers)
    weights = network.get_weights()
    layer_activities =  network.forward_pass(data)  # List of matrices of unit activities. Contains l-matrices, where l= numLayers. Order is bottom-up.
                                                    # Every matrix is m x n, where m= numTrainingExamples, n= num units in that layer   

    dE_dY = layer_activities[num_layers -1] - targets #dE/dy for output layer
 
    for j in range(num_layers-1, 0, -1):
        dY_dZ = layers[j].gradient() #get dy/dz from the layer's gradient method
        dE_dZ = dY_dZ * dE_dY
        
        dE_dY = dot(dE_dZ, weights[j-1].transpose()) #This will be dE/dY used in next layer

        dE_dW = dot(layer_activities[j-1].transpose(), dE_dZ)/data.shape[0] #Normalised dE/dW matrix for this layer over all training examples
        
        network_dE_dW.insert(0, dE_dW) #Put this at the front of the list of network deltas 

    return network_dE_dW
        
def dE_dW(X, Y, dE_dY):
    N = X.shape[0]
    Y_size, X_size = Y.shape[1], X.shape[1]
    dY_dZ = Y*(1-Y)
    assert dY_dZ.shape == (N, Y_size), "dY_dZ shape is wrong"
    dZ_dW = X
    assert dZ_dW.shape == (N, X_size), "dZ_dW shape is wrong"
    dE_dZ = dE_dY * dY_dZ
    assert dE_dZ.shape == (N, Y_size), "dE_dZ shape is wrong"
    dE_dB = (dE_dZ.sum(0)/N).reshape((1, Y.shape[1]))
    dE_dW = dot(transpose(dZ_dW), dE_dZ)/N
    assert dE_dW.shape == (X_size, Y_size), "dE_dW shape is wrong"

    return dE_dW, dE_dB, dE_dZ

def print_squared_error(T, Y):
    print 'Squared Error:', square(T-Y).sum()

def print_cross_entropy(T, Y):
    print 'Cross-Entropy Error:', -(T*log(Y)).sum()
def dcost_entropy(T, Y):
    return -T/Y

def train_net(net, X, T, dCfunc=dcost_entropy, learning_rate=0.1):
    layer_activities = net.forward_pass(X)
    print_cross_entropy(T, layer_activities[-1])
    nw = len(net.weights)
    i = nw

    dE_dlayer = dCfunc(T,layer_activities[-1])

    weight_derivatives = []
    bias_derivatives = []
    while i > 0:
        dEdW, dEdB, dEdZ = dE_dW(layer_activities[i-1], layer_activities[i], dE_dlayer)
        weight_derivatives.append(dEdW)
        bias_derivatives.append(dEdB)
        dE_dlayer = dot(dEdZ, transpose(net.weights[i-1]))

        i -= 1
    print weight_derivatives
    i = nw
    for delta_weights in weight_derivatives:
        net.weights[i-1] -= learning_rate * delta_weights
        i -= 1
    i = nw
    for delta_bias in bias_derivatives:
        net.layers[i].bias -= learning_rate * delta_bias
        i -=1
