from numpy import *
from network import *
from scipy.io import loadmat
import pylab
import backprop

def learn_curve(network, training_input, training_target, validation_input, validation_target, epochs=50):
    '''Plots the erros w.r.t the number of epochs of training. Assumes no regularisation.
    Returns two lists containing the training and validation errors.'''
    
    training_errors = []
    validation_errors = []

    num_training = len(training_input)
    
    #Train mini batches of 10s
    minibatches = [(training_input[i:i+10, :], training_target[i:i+10, :]) for i in range(0, num_training, 10)]
    for epoch in range(epochs):
        for batch in minibatches:
            backprop.train(network, batch[0], batch[1])

        #Get errors for this epoch
        train_error = error(network, training_input, training_target)
        val_error = error(network, validation_input, validation_target)
        training_errors.append(train_error)
        validation_errors.append(val_error)

    #Plot the data
    pylab.plot(training_errors, label="Training Error")
    pylab.plot(validation_errors, label="Validation Error")
    pylab.title("Training and Validation Errors w.r.t. Number of Epochs")
    pylab.xlabel("Epochs")
    pylab.ylabel("Error")
    pylab.legend()
    pylab.show()

    return training_errors, validation_errors
    
def vary_data(network, training_input, training_target, validation_input, validation_target, epochs=5):
    '''Plots training and validation errors of a net w.r.t. the number of training
    cases used in learning. Useful for determining if the issue is in the amount of data
    Returns two lists containing the training and validation errors'''
    training_errors = []
    validation_errors = []

    num_training = len(training_input)
    
    for percent in range(5, 105, 5): #Sweep through percentages from 5 to 100%
        train_length = num_training*percent/100.0
        validation_length = len(validation_input)*percent/100.0
        network_copy = network
    #Train mini batches of 10s
        minibatches = [(training_input[i:i+10, :], training_target[i:i+10, :]) for i in range(0, int(train_length), 10)]
        for epoch in range(epochs):
            for batch in minibatches:
                backprop.train(network_copy, batch[0], batch[1])

        #Get errors for this percent of data
        train_error = error(network_copy, training_input[0:train_length], training_target[0:train_length])
        val_error = error(network_copy, validation_input[0:validation_length], validation_target[0:validation_length])
        training_errors.append(train_error)
        validation_errors.append(val_error)

    #Plot the data
    pylab.plot(range(5,105,5), training_errors, label="Training Error")
    pylab.plot(range(5,105,5), validation_errors, label="Validation Error")
    pylab.title("Training and Validation Errors w.r.t. % of Data used")
    pylab.xlabel("Data Used (%)")
    pylab.ylabel("Error After "+ str(epochs)+ " Epochs")
    pylab.legend()
    pylab.show()

    return training_errors, validation_errors
    
def reg_parameter(network, training_input, training_target, validation_input, validation_target, sweep_range=(0.01, 20), epochs=10):
    '''Plots training and validation errors of a net varying the regularisation parameter over a set range. Useful for determining 
    where network is underfitting and overfitting. Sweep Range is a tuple, where the first value is the initial decay rate to test, 
    and the second value is the number of parameters to test. Each parameter will be 1.2 times the one before it.
    Returns two lists containing the training and validation errors'''
    training_errors = []
    validation_errors = []

    num_training = len(training_input)
    start_rate = sweep_range[0]
    num_parameters = sweep_range[1]
    decay_rates = [start_rate*(1.2**i) for i in range(num_parameters)] #Create a list of different regularisation parameters to try.
    decay_rates.insert(0,0)
    for rate in decay_rates:
        #Train mini batches of 10s
        new_net = network
        minibatches = [(training_input[i:i+10, :], training_target[i:i+10, :]) for i in range(0, num_training, 10)]
        for epoch in range(epochs):
            for batch in minibatches:
                backprop.train(new_net, batch[0], batch[1], rate)

        #Get errors for this decay rate
        train_error = error(new_net, training_input, training_target, rate)
        val_error = error(new_net, validation_input, validation_target, rate)
        training_errors.append(train_error)
        validation_errors.append(val_error)

    #Plot the data
    pylab.plot(decay_rates, training_errors, label="Training Error")
    pylab.plot(decay_rates,validation_errors, label="Validation Error")
    pylab.title("Training and Validation Errors w.r.t. Regularisation Parameter")
    pylab.xlabel("Regularisation Parameter")
    pylab.ylabel("Error After 10 Epochs")
    pylab.legend()
    pylab.show()

    return training_errors, validation_errors

def learn_rate(network, data):
    '''Plots training and validation errors of a net when varying the learning rate. Useful
    for finding a good learning rate for the net.'''

def vary_hidden(network, data, sweep_range):
    '''Plots training and validation errors of a net for different sized hidden layers in the
    specified range'''

def vary_numlayers(network, data, sweep_range):
    '''Plots training and validation errors of a net by adding additional hidden layers of equal
    size, in the specified sweep range.'''

def vary_sigmoid(network, training_input, training_target, validation_input, validation_target, epochs=50):
    '''Plots errors for a full sweep of varying the hidden layers to different types of sigmoid
    units.'''

def error(net, data, T, decay_rate=0):
    '''Cross-entropy error of a net'''
    Y = net.forward_pass(data)[-1]
    weights = net.get_weights()
    error = -(T*log(Y) + (1-T)*log(1-Y)).sum()  # Unregularised error term

    regerror = 0
    for layer in weights:
        regerror += (layer*layer).sum()  # Add the square of every weight in the network
    regerror *= (decay_rate/2)  # Normalise by decay constant
    return (error + regerror)/len(data) #Error normalised by number of training examples

def gradcheck(network, layer, data, targets):
    '''Checks gradients obtained by backpropagation by getting gradients numerically'''
    dE_dW = []
    epsilon = 0.00001

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

            grad = ((error1 - error2)/(2*epsilon))
            dE_dW.append(grad)

            wmatrix[i][j] += epsilon
            temp_weights[layer] = wmatrix
            network.set_weights(temp_weights)

    network.set_weights(weights)  # Back to original weights
    return dE_dW
