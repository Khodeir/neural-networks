from numpy import *
from numpy.matlib import repmat


class Layer(object):
    def __init__(self, size, bias=None, activities=None, dropoutrate=0):
        self.size = size
        self.bias = bias if bias is not None else zeros((1, size))
        self.activities = activities if activities is not None else zeros((1, size))
        self.dropoutrate = dropoutrate

    @classmethod
    def from_layer(cls, layer):
        return cls(layer.size, layer.bias.copy(), layer.activities.copy())

    def switch_type(self, newlayertype):
        '''Switch the layer to the given newlayertype'''
        self.__class__ = newlayertype

    def process(self, weighted_input):
        self.activities = dropout(self.act(weighted_input), self.dropoutrate)

    def act(self, weighted_input):
        assert False, "This is an abstract class"

    def gradient(self):
        '''I'm having every layer compute it's own dy/dz 's, that way backprop can work with any layer
        type, where the layer provides backprop with dy/dz, using it's activity values '''
        assert False, "This is an abstract class"

    def repbias(self, data):
        '''Replicates the bias vector in so that it can be used in matrix operations with data'''
        return repmat(self.bias, data.shape[0], 1)


class LogisticLayer(Layer):
    def act(self, weighted_input):
        return 1/(1 + exp(-(weighted_input + self.repbias(weighted_input))))

    def gradient(self):
        #In logistic units, dy/dz = y*(1-y)
        return self.activities * (1 - self.activities)

class TanhLayer(Layer):
    '''A fast changing logistic function'''
    def act(self, weighted_input):
        return tanh(weighted_input + self.repbias(weighted_input))

    def gradient(self):
        #dy/dz = 1-y^2
        return (1 - (self.activities*self.activities))

class BinaryStochasticLayer(LogisticLayer):
    def __init__(self, size, bias=None, activities=None):
        LogisticLayer.__init__(self, size, bias, activities)
        self.probs = self.activities.copy()
    def act(self, weighted_input):
        self.probs = LogisticLayer.act(self, weighted_input)
        return sample_binary_stochastic(self.probs)
    def gradient(self):
        #Use probs instead of activities, dy/dz = y*(1-y). So that backprop works with BinaryStochastic
        return self.probs * (1 - self.probs)

class LinearLayer(Layer):
    def act(self, weighted_input):
        return weighted_input + self.repbias(weighted_input)
    def gradient(self):
        return 1
        
class LinearThresholdLayer(Layer):
    def act(self, weighted_input):
        activity = weighted_input + self.repbias(weighted_input)
        return activity*(activity >= 0) #If activity is negative, output a 0

class BinaryThresholdLayer(Layer):
    def act(self, weighted_input):
        return ((weighted_input + self.repbias(weighted_input)) > zeros((1, self.size))).astype(int)


class SoftMax(Layer):
    def normalizer(self, a):
        max_small = a.max(axis=1)
        max_big = repmat(max_small, a.shape[1], 1).transpose()
        return log(exp(a - max_big).sum(1)) + max_small

    def act(self, weighted_input):
        normalizer = self.normalizer(weighted_input).reshape((1, weighted_input.shape[0]))
        log_prob = weighted_input - repmat(normalizer, weighted_input.shape[1], 1).transpose()
        return exp(log_prob)

    def gradient(self):
        return self.activities * (1 - self.activities)

class HybridLayer(SoftMax, BinaryStochasticLayer):
    '''Assumes segmentA is softmax and segmentB is BinaryStochastic'''
    def __init__(self, sizeA, sizeB):
        self.sizeA = sizeA
        Layer.__init__(self, sizeA + sizeB)
        self.probs = self.activities.copy()
        self.activitiesA = zeros((1, sizeA))

    def act(self, weighted_input):
        actA = SoftMax.act(self, weighted_input[:,0:self.sizeA])
        self.activitiesA = (actA - actA.max(1).reshape((weighted_input.shape[0], 1)) == 0).astype(int)
        self.probs = 1/(1 + exp(-(weighted_input[:,self.sizeA:] + self.repbias(weighted_input[:,self.sizeA:], self.sizeA))))
        self.probs = concatenate((self.activitiesA, self.probs), axis=1)
        
        return concatenate((self.activitiesA, sample_binary_stochastic(self.probs[:,self.sizeA:])), axis=1)

    def repbias(self, data, startIndex=0):
        '''Replicates the bias vector in so that it can be used in matrix operations with data'''
        return repmat(self.bias[:,startIndex:   ], data.shape[0], 1)

def sample_binary_stochastic(probmat):
    return (probmat > random.random(probmat.shape)).astype(int)
def dropout(data, rate=0.2):
    if rate == 0:
        return data
    drop = random.binomial(1, rate, data.shape)
    return data - data * drop