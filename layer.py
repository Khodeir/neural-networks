from numpy import *
from numpy.matlib import repmat

class Layer(object):
    def __init__(self, size):
        self.size = size
        self.bias = zeros((1, size))
    def process(self, weighted_input):
        assert False, "This is an abstract class"
    def repbias(self, data):
        '''Replicates the bias vector in so that it can be used in matrix operations with data'''
        return repmat(self.bias, data.shape[0], 1)

class LogisticLayer(Layer):
    def process(self, weighted_input):
        return 1/(1 + exp(-(weighted_input + self.repbias(weighted_input))))

class LinearLayer(Layer):
    def process(self, weighted_input):
        return weighted_input + self.repbias(weighted_input)

class BinaryThresholdLayer(Layer):
    def process(self, weighted_input):
        return ((weighted_input + self.repbias(weighted_input)) > zeros((1, self.size))).astype(int)

class SoftMax(Layer):
    def process(self, weighted_input):
        Z = weighted_input + self.repbias(weighted_input)
        reg = transpose(repmat(exp(Z).sum(1), self.size, 1))
        return exp(Z)/reg