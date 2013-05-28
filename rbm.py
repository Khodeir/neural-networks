from numpy import *
from numpy.matlib import repmat
from layer import LogisticLayer
import matplotlib.pyplot as plt


class RBM(object):
    def __init__(self, numvis, numhid):
        '''Initialize an RBM with numvis visible units and numhid hidden units. The weights are randomly initialized.'''
        self.numvis = numvis
        self.numhid = numhid
        self.vislayer = LogisticLayer(numvis)
        self.hidlayer = LogisticLayer(numhid)
        self.vishid = 0.03*random.randn(numvis, numhid)

    def sample_hid(self, data, binary_stochastic=False, dropout=False):
        if dropout:
            drop = (random.random((data.shape)) > random.random((data.shape))).astype(int)
            data -= drop*data
        hidinput = dot(data, self.vishid)
        hidprob = self.hidlayer.process(hidinput)
        if binary_stochastic:
            return sample_binary_stochastic(hidprob)
        return hidprob

    def sample_vis(self, data, binary_stochastic=False, dropout=True):
        if dropout:
            drop = (random.random((data.shape)) > random.random((data.shape))).astype(int)
            data -= drop*data
        visinput = dot(data, transpose(self.vishid))
        visprob = self.vislayer.process(visinput)
        if binary_stochastic:
            return sample_binary_stochastic(visprob)
        return visprob

    def train(self, data, K, epochs, learning_rate=0.1, weightcost=0):
        '''Train the network using normalized data and CD-k for epochs epochs'''
        assert self.numvis == data.shape[1], "Data does not match number of visible units."
        #got to initialize some vars
        delta_vishid = zeros((self.numvis, self.numhid))
        delta_bias_vis = zeros((1, self.numvis))
        delta_bias_hid = zeros((1, self.numhid))
        momentum = 0.9

  #start epochs
        for epoch in range(epochs):
            #get the acivation probabilities for hidden units on each data case
            hidprobs_data = self.sample_hid(data)  # NxH matrix

            #compute the positive term in cd learning rule, Expected(sisj)_data
            expect_pairact_data = dot(transpose(data), hidprobs_data)

            #The same quantitiy for our biases, Expected(si)_data (i.e. bias unit is always 1)
            expect_bias_hid_data = hidprobs_data.sum(0)
            expect_bias_vis_data = data.sum(0)

            #now we start the contrastive divergence using stochastic binary hidstates
            visprobs_cd = data
            hidprobs_cd = hidprobs_data
            for k in range(K):
                hidstates = sample_binary_stochastic(hidprobs_cd)
                visprobs_cd = self.sample_vis(hidstates)
                hidprobs_cd = self.sample_hid(visprobs_cd)

            #now we compute the negative statistics for contrastive divergence, Expected(sisj)_model
            expect_pairact_cd = dot(transpose(visprobs_cd), hidprobs_cd)

            #again negative stats for learning the biases
            expect_bias_hid_cd = hidprobs_cd.sum(0)
            expect_bias_vis_cd = visprobs_cd.sum(0)

            recons_error = square(data - visprobs_cd).sum()

            #learning time

            N = float(data.shape[0])
            delta_vishid *= momentum
            delta_vishid += (learning_rate/N)*((expect_pairact_data - expect_pairact_cd) - weightcost*self.vishid)

            delta_bias_vis *= momentum
            delta_bias_vis += (learning_rate/N)*(expect_bias_vis_data - expect_bias_vis_cd)
            delta_bias_hid *= momentum
            delta_bias_hid += (learning_rate/N)*(expect_bias_hid_data - expect_bias_hid_cd)

            self.vishid += delta_vishid
            self.vislayer.bias += delta_bias_vis
            self.hidlayer.bias += delta_bias_hid

            print 'Reconstruction Error:', recons_error

    def feature_map(self, (featN, featM), (mapN, mapM)):
        assert self.numvis == featN * featM, 'Number of visible units must not change'
        vh = transpose(self.vishid)
        result = zeros((featN*mapN, featM*mapM))
        row = 0
        col = 0
        for i in range(self.numhid):
            result[row:row+featN, col:col+featM] = vh[i].reshape((featN, featM))
            col += featM
            if col == featM*mapM:
                row += featN
                col = 0
        return result


def draw(data):
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.show()


def sample_binary_stochastic(probmat):
    return (probmat > random.random(probmat.shape)).astype(int)
