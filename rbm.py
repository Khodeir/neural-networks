from layer import *
from network import NeuralNet, sample_binary_stochastic, dropout
from numpy import *


class RBM(NeuralNet):
    def __init__(self, numvis, numhid, vislayer=None, hidlayer=None, vishid=None):
        '''Initialize an RBM with numvis visible units and numhid hidden units. The weights are randomly initialized
        explicitly passed in as a parameter.'''
        self.numvis = numvis
        self.numhid = numhid
        weights = [vishid] if vishid is not None else None
        NeuralNet.__init__(self, [vislayer or BinaryStochasticLayer(numvis), hidlayer or BinaryStochasticLayer(numhid)], weights)

    def get_vislayer(self):
        return self.layers[0]

    def get_hidlayer(self):
        return self.layers[1]

    def get_vishid(self):
        return self.weights[0]

    def sample_hid(self, data, prob=False):
        '''Samples the hidden layer of the rbm given the parameter data as the state of the visibles'''
        data = self.forward_pass(data, skip_layer=1)[1]
        return self.get_hidlayer().probs if prob else data

    def sample_vis(self, data, prob=False):
        '''Samples the visible layer of the rbm given the parameter data as the state of the hiddens'''
        data = self.backward_pass(data, skip_layer=1)[0]
        return self.get_vislayer().probs if prob else data

    def gibbs_given_h(self, data, K, dropoutrate=0):
        '''Performs K steps back and forth between hidden and visible starting from the parameter data as the state of the hiddens.
        data is assumed to be the current activation of h.'''
        hidact = data
        visact = None
        for k in range(K):
            visact = self.sample_vis(dropout(hidact, dropoutrate))
            hidact = self.sample_hid(dropout(visact, dropoutrate))
        return visact, hidact

    def gibbs_given_v(self, data, K, dropoutrate=0):
        '''Performs K steps back and forth between visible and hidden starting from the parameter data as the state of the visibles.
        data is assumed to be the current activation of v'''
        visact = data
        for k in range(K):
            hidact = self.sample_hid(dropout(visact, dropoutrate))
            visact = self.sample_vis(dropout(hidact, dropoutrate))
        return visact, hidact

    def reconstruction_error(self, data, K=1):
        self.gibbs_given_v(data, K)
        visprobs = self.get_vislayer().probs
        return square(data - visprobs).sum()

    def train(self, data, K, learning_rate=0.1, weightcost=0.0001, dropoutrate=0):
        '''Train the network using normalized data and CD-K for epochs epochs'''
        assert self.numvis == data.shape[1], "Data does not match number of visible units."
        #got to initialize some vars
        delta_vishid = zeros((self.numvis, self.numhid))
        delta_bias_vis = zeros((1, self.numvis))
        delta_bias_hid = zeros((1, self.numhid))

        #get the acivation probabilities for hidden units on each data case
        self.get_vislayer().probs = data
        hidact_data = self.sample_hid(data)  # NxH matrix
        hidprobs_data = self.get_hidlayer().probs

        #compute the positive term in cd learning rule, Expected(sisj)_data
        expect_pairact_data = dot(transpose(data), hidprobs_data)

        #The same quantitiy for our biases, Expected(si)_data (i.e. bias unit is always 1)
        expect_bias_hid_data = hidprobs_data.sum(0)
        expect_bias_vis_data = data.sum(0)

        #now we get the logistic output after K steps of gibbs sampling and use that as probability of turning on
        self.gibbs_given_h(hidact_data, K, dropoutrate)
        visprobs_cd, hidprobs_cd = self.get_vislayer().probs, self.get_hidlayer().probs

        #now we compute the negative statistics for contrastive divergence, Expected(sisj)_model
        expect_pairact_cd = dot(transpose(visprobs_cd), hidprobs_cd)

        #again negative stats for learning the biases
        expect_bias_hid_cd = hidprobs_cd.sum(0)
        expect_bias_vis_cd = visprobs_cd.sum(0)

        recons_error = square(data - visprobs_cd).sum()

        #learning time

        N = float(data.shape[0])
        delta_vishid += (learning_rate/N)*((expect_pairact_data - expect_pairact_cd) - weightcost*self.weights[0])
        # delta_bias_vis += (learning_rate/N)*(expect_bias_vis_data - expect_bias_vis_cd)
        delta_bias_hid += (learning_rate/N)*(expect_bias_hid_data - expect_bias_hid_cd)

        self.weights[0] += delta_vishid
        self.layers[0].bias += delta_bias_vis
        self.layers[1].bias += delta_bias_hid

        #print 'Reconstruction Error:', recons_error
        return recons_error
