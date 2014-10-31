from numpy import *
from numpy.matlib import repmat


class Layer(object):
    def __init__(self, size, bias=None, activities=None):
        self.size = size
        self.bias = bias if bias is not None else zeros((1, size))
        self.activities = activities if activities is not None else zeros((1, size))

    @classmethod
    def from_layer(cls, layer):
        return cls(layer.size, layer.bias.copy(), layer.activities.copy())

    def switch_type(self, newlayertype):
        '''Switch the layer to the given newlayertype'''
        self.__class__ = newlayertype

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
        return self.activities * (1 - self.activities)

class TanhLayer(Layer):
    '''A fast changing logistic function'''
    def process(self, weighted_input):
        self.activities = tanh(weighted_input + self.repbias(weighted_input))
        return self.activities

    def gradient(self):
        #dy/dz = 1-y^2
        return (1 - (self.activities*self.activities))

class BinaryStochasticLayer(LogisticLayer):
    def __init__(self, size, bias=None, activities=None):
        LogisticLayer.__init__(self, size, bias, activities)
        self.probs = self.activities.copy()
    def process(self, weighted_input):
        self.probs = LogisticLayer.process(self, weighted_input)
        self.activities = sample_binary_stochastic(self.probs)
        return self.activities

class LinearLayer(Layer):
    def process(self, weighted_input):
        self.activities = weighted_input + self.repbias(weighted_input)
        return self.activities

class LinearThresholdLayer(Layer):
    def process(self, weighted_input):
        activity = weighted_input + self.repbias(weighted_input)
        self.activities = activity*(activity >= 0) #If activity is negative, output a 0
        return self.activities

class BinaryThresholdLayer(Layer):
    def process(self, weighted_input):
        self.activities = ((weighted_input + self.repbias(weighted_input)) > zeros((1, self.size))).astype(int)
        return self.activities


class SoftMax(Layer):
    def normalizer(self, a):
        max_small = a.max(axis=1)
        max_big = repmat(max_small, a.shape[1], 1).transpose()
        return log(exp(a - max_big).sum(1)) + max_small

    def process(self, weighted_input):
        normalizer = self.normalizer(weighted_input).reshape((1, weighted_input.shape[0]))
        log_prob = weighted_input - repmat(normalizer, weighted_input.shape[1], 1).transpose()
        self.activities = exp(log_prob)
        return self.activities

    def gradient(self):
        return self.activities * (1 - self.activities)

class HybridLayer(SoftMax, BinaryStochasticLayer):
    '''Assumes segmentA is softmax and segmentB is BinaryStochastic'''
    def __init__(self, sizeA, sizeB):
        self.sizeA = sizeA
        Layer.__init__(self, sizeA + sizeB)
        self.probs = self.activities.copy()
        self.activitiesA = zeros((1, sizeA))

    def process(self, weighted_input):
        SoftMax.process(self, weighted_input[:,0:self.sizeA])
        self.activitiesA = (self.activities - self.activities.max(1).reshape((weighted_input.shape[0], 1)) == 0).astype(int)
        self.probs = 1/(1 + exp(-(weighted_input[:,self.sizeA:] + self.repbias(weighted_input[:,self.sizeA:], self.sizeA))))
        self.probs = concatenate((self.activities, self.probs), axis=1)
        self.activities = concatenate((self.activitiesA, sample_binary_stochastic(self.probs[:,self.sizeA:])), axis=1)

    def repbias(self, data, startIndex=0):
        '''Replicates the bias vector in so that it can be used in matrix operations with data'''
        return repmat(self.bias[:,startIndex:   ], data.shape[0], 1)

def sample_binary_stochastic(probmat):
    return (probmat > random.random(probmat.shape)).astype(int)
