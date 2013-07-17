from metrics import error
from backprop import flat_grad
from layer import *
from network import *
from scipy.optimize import minimize

def cross_err(x, network, data, targets):

	network.set_parameters(x)

	return error(network, data, targets, decay_rate=0)

def squared_err(x, network, data, targets):

	network.set_parameters(x)

	return square(network.forward_pass(data)[-1] - targets).sum()

def deriv(x, network, data, targets):
	'''Network error derivatives using backprop. x is a single vector with all of a network's parameters.'''

	network.set_parameters(x)

	return flat_grad(network, data, targets)

def BFGS(network, initial, data, targets, maxiter=1000):
	weights = minimize(cross_err, initial, [(network), (data), (targets)], method='BFGS', jac = deriv, options={'disp':True, 'maxiter':maxiter})
	return weights

def L_BFGS(network, initial, data, targets, maxiter=1000, bounds=None):
	weights = minimize(cross_err, initial, [(network), (data), (targets)], method='L-BFGS-B', jac = deriv, bounds=bounds, options={'disp':True, 'maxiter':maxiter})
	return weights

def CG(network, initial, data, targets, maxiter=1000):
	weights = minimize(cross_err, initial, [(network), (data), (targets)], method='CG', jac = deriv, options={'disp':True, 'maxiter':maxiter})
	return weights

def Newton_CG(network, initial, data, targets, maxiter=1000):
	weights = minimize(cross_err, initial, [(network), (data), (targets)], method='Newton-CG', jac = deriv, options={'disp':True, 'maxiter':maxiter})
	return weights
