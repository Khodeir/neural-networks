from numpy import *
from numpy.matlib import repmat

class Layer(object):
    def __init__(self, size):
        self.size = size
        self.bias = zeros((1, size))
        self.activities = zeros((1, size))
        self.grad = zeros((1,size))
    def process(self, weighted_input):
        assert False, "This is an abstract class"
    def gradient(self):
        '''I'm having every layer compute it's own dy/dz 's, that way backprop can work with any layer
        type, where the layer provides backprop with dy/dz, using it's activity values '''
        assert False, "This is an abstract class"
    def repbias(self, data):
        '''Replicates the bias vector in so that it can be used in matrix operations with data'''
        return repmat(self.bias, data.shape[0], 1)

class LogisticLayer(Layer):
    def process(self, weighted_input):
        self.activities = 1/(1 + exp(-(weighted_input + self.repbias(weighted_input))))
        return self.activities
    def gradient(self):
        #In logistic units, dy/dz = y*(1-y)
        self.grad = self.activities * (1- self.activities)
        return self.grad

#I still need to add gradient methods for all the other kinds of units
class LinearLayer(Layer):
    def process(self, weighted_input):
        self.activities = weighted_input + self.repbias(weighted_input)
        return self.activities

class BinaryThresholdLayer(Layer):
    def process(self, weighted_input):
        self.activities = ((weighted_input + self.repbias(weighted_input)) > zeros((1, self.size))).astype(int)
        return self.activities

class SoftMax(Layer):
    def process(self, weighted_input):
        Z = weighted_input + self.repbias(weighted_input)
        reg = transpose(repmat(exp(Z).sum(1), self.size, 1))
        return exp(Z)/reg
