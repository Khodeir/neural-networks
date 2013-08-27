from metrics import error
from backprop import backprop, dE_cross_entropy
from layer import *
from network import *
from scipy.optimize import minimize

def err(x, network, data, targets, error_func=error, dE_func=dE_cross_entropy):

    network.set_parameters(x)

    return error_func(network, data, targets, tau=0)

def squared_err(x, network, data, targets):

    network.set_parameters(x)

    return square(network.forward_pass(data)[-1] - targets).sum()

def deriv(x, network, data, targets, error_func=error, dE_func=dE_cross_entropy):
    '''Network error derivatives using backprop. x is a single vector with all of a network's parameters.'''

    network.set_parameters(x)
    a,b = backprop(network, data, targets, tau=0, dE_func= dE_func)
    flat_grad = []
    
    for matrix in a:
        flat_grad = concatenate((flat_grad, matrix.flatten()),1)
    for matrix in b:
        flat_grad = concatenate((flat_grad, matrix.flatten()),1)

    return flat_grad

def BFGS(network, data, targets, maxiter=1000, error_func=error, dE_func=dE_cross_entropy):
    initial = network.flatten_parameters()
    weights = minimize(err, initial, [(network),(data),(targets),(error_func),(dE_func)], method='BFGS', jac = deriv, options={'disp':True, 'maxiter':maxiter})
    return weights

def L_BFGS(network, data, targets, maxiter=1000, error_func=error, dE_func=dE_cross_entropy, bounds=None):
    initial = network.flatten_parameters()
    weights = minimize(err, initial, [(network),(data),(targets),(error_func),(dE_func)], method='L-BFGS-B', jac = deriv, bounds=bounds, options={'disp':True, 'maxiter':maxiter})
    return weights

def CG(network, data, targets, maxiter=25, error_func=error, dE_func=dE_cross_entropy):
    initial = network.flatten_parameters()
    weights = minimize(err, initial, [(network),(data),(targets),(error_func),(dE_func)], method='CG', jac = deriv, options={'disp':True, 'maxiter':maxiter})
    return weights

def Newton_CG(network, data, targets, maxiter=1000, error_func=error, dE_func=dE_cross_entropy):
    initial = network.flatten_parameters()
    weights = minimize(err, initial, [(network),(data),(targets),(error_func),(dE_func)], method='Newton-CG', jac = deriv, options={'disp':True, 'maxiter':maxiter})
    return weights
