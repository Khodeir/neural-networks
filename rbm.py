from layer import LogisticLayer
from network import NeuralNet, sample_binary_stochastic
from numpy import *



class RBM(NeuralNet):
    def __init__(self, numvis, numhid):
        '''Initialize an RBM with numvis visible units and numhid hidden units. The weights are randomly initialized.'''
        self.numvis = numvis
        self.numhid = numhid
        NeuralNet.__init__(self, [LogisticLayer(numvis), LogisticLayer(numhid)])

    def sample_hid(self, data, binary_stochastic=False, dropout=False):
        if dropout:
            drop = (random.random((data.shape)) > random.random((data.shape))).astype(int)
            data -= drop*data
        hidinput = dot(data, self.weights[0])
        hidprob = self.layers[1].process(hidinput)
        if binary_stochastic:
            return sample_binary_stochastic(hidprob)
        return hidprob

    def sample_vis(self, data, binary_stochastic=False, dropout=True):
        if dropout:
            drop = (random.random((data.shape)) > random.random((data.shape))).astype(int)
            data -= drop*data
        visinput = dot(data, transpose(self.weights[0]))
        visprob = self.layers[0].process(visinput)
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
            delta_vishid += (learning_rate/N)*((expect_pairact_data - expect_pairact_cd) - weightcost*self.weights[0])

            delta_bias_vis *= momentum
            delta_bias_vis += (learning_rate/N)*(expect_bias_vis_data - expect_bias_vis_cd)
            delta_bias_hid *= momentum
            delta_bias_hid += (learning_rate/N)*(expect_bias_hid_data - expect_bias_hid_cd)

            self.weights[0] += delta_vishid
            self.layers[0].bias += delta_bias_vis
            self.layers[1].bias += delta_bias_hid

            print 'Reconstruction Error:', recons_error
