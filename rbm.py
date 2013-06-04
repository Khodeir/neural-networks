from layer import LogisticLayer
from network import NeuralNet, sample_binary_stochastic, dropout
from numpy import *


class RBM(NeuralNet):
    def __init__(self, numvis, numhid, vislayer=None, hidlayer=None, vishid=None):
        '''Initialize an RBM with numvis visible units and numhid hidden units. The weights are randomly initialized
        explicitly passed in as a parameter.'''
        self.numvis = numvis
        self.numhid = numhid
        weights = [vishid] if vishid is not None else None
        NeuralNet.__init__(self, [vislayer or LogisticLayer(numvis), hidlayer or LogisticLayer(numhid)], weights)

    def get_vislayer(self):
        return self.layers[0]

    def get_hidlayer(self):
        return self.layers[1]

    def get_vishid(self):
        return self.weights[0]

    def sample_hid(self, data):
        '''Samples the hidden layer of the rbm given the parameter data as the state of the visibles'''
        hidprob = self.forward_pass(data, skip_layer=1)[1]
        return hidprob

    def sample_vis(self, data):
        '''Samples the visible layer of the rbm given the parameter data as the state of the hiddens'''
        visprob = self.backward_pass(data, skip_layer=1)[0]
        return visprob

    def gibbs_given_h(self, data, K, dropoutrate=0):
        '''Performs K steps back and forth between hidden and visible starting from the parameter data as the state of the hiddens'''
        for k in range(K):
            hidstates = sample_binary_stochastic(data)
            visprobs_cd = self.sample_vis(dropout(hidstates, dropoutrate))
            visstates = sample_binary_stochastic(visprobs_cd)
            hidprobs_cd = self.sample_hid(dropout(visstates, dropoutrate))
        return visprobs_cd, hidprobs_cd

    def gibbs_given_v(self, data, K, dropoutrate=0):
        '''Performs K steps back and forth between visible and hidden starting from the parameter data as the state of the visibles'''
        for k in range(K):
            visstates = sample_binary_stochastic(data)
            hidprobs_cd = self.sample_hid(dropout(visstates, dropoutrate))
            hidstates = sample_binary_stochastic(visprobs_cd)
            visprobs_cd = self.sample_vis(dropout(hidstates, dropoutrate))
        return visprobs_cd, hidprobs_cd

    def train(self, data, K, epochs, learning_rate=0.1, weightcost=0.1, momentum=0.7, dropoutrate=0):
        '''Train the network using normalized data and CD-K for epochs epochs'''
        assert self.numvis == data.shape[1], "Data does not match number of visible units."
        #got to initialize some vars
        delta_vishid = zeros((self.numvis, self.numhid))
        delta_bias_vis = zeros((1, self.numvis))
        delta_bias_hid = zeros((1, self.numhid))

  #start epochs
        for epoch in range(epochs):
            #get the acivation probabilities for hidden units on each data case
            hidprobs_data = self.sample_hid(data)  # NxH matrix

            #compute the positive term in cd learning rule, Expected(sisj)_data
            expect_pairact_data = dot(transpose(data), hidprobs_data)

            #The same quantitiy for our biases, Expected(si)_data (i.e. bias unit is always 1)
            expect_bias_hid_data = hidprobs_data.sum(0)
            expect_bias_vis_data = data.sum(0)

            #now we get the logistic output after K steps of gibbs sampling and use that as probability of turning on
            visprobs_cd, hidprobs_cd = self.gibbs_given_h(hidprobs_data, K, dropoutrate)

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

            # delta_bias_vis *= momentum
            # delta_bias_vis += (learning_rate/N)*(expect_bias_vis_data - expect_bias_vis_cd)
            delta_bias_hid *= momentum
            delta_bias_hid += (learning_rate/N)*(expect_bias_hid_data - expect_bias_hid_cd)

            self.weights[0] += delta_vishid
            self.layers[0].bias += delta_bias_vis
            self.layers[1].bias += delta_bias_hid

            print 'Reconstruction Error:', recons_error
