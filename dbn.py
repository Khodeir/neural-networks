from network import NeuralNet
from layer import *


class BN(object):
    '''A BN is a bidirectional NeuralNet. It is equivelant to two opposite direction feed forward nets.'''
    def __init__(self, layers, up_weights=None, down_weights=None):
        '''Initializes a BN from the layers given'''
        self.numlayers = len(layers)
        self.upnet = NeuralNet(layers, up_weights)
        self.downnet = NeuralNet(layers[::-1], down_weights)

    @classmethod
    def from_rbms(cls, rbms):
        '''Initializes a BN from a list of RBMS. NOTE: the down weights and upweights are tied.
        Modifying one, modifies the other. To untie, call the __untie__ method.'''
        layers = []

        # First layer of dbn is the visible layer of the bottom rbm
        layers.append(rbms[0].get_vislayer())
        # Keep all hidden layers
        for rbm in rbms:
            layers.append(rbm.get_hidlayer())

        up_weights = [rbm.get_vishid() for rbm in rbms]
        down_weights = [rbm.get_vishid().transpose() for rbm in rbms[::-1]]

        return cls(layers, up_weights, down_weights)

    def bottom_up(self, data):
        '''Expects data to be probabilities'''
        self.upnet.layers[0].probs = data
        self.upnet.layers[0].activities = sample_binary_stochastic(data)
        return self.upnet.forward_pass(data, 1)

    def top_down(self, data):
        '''Expects data to be binary'''
        self.downnet.layers[0].activities = data
        return self.downnet.forward_pass(data, 1)


    def __untie_weights__(self):
        '''This is an ugly step, and is only necessary when the db is initialized from RBMs.
        It unties the recognition weights from the generative ones.'''
        numweights = self.numlayers - 1
        for i in range(numweights):
            self.downnet.weights[i] = self.downnet.weights[i].copy()

    def wake_phase(self, data):
        '''The first step of wake-sleep and contrastive wake-sleep. Returns wake_deltas, a list of matrices by which the
        the weights of the down net should be adjusted. Also returns hidden states of top layer. Assumes DBN layers are 
        binary stochastic layers.'''
        #Get the states and probabilities of every layer after doing a bottom-up pass
        hid_states = self.bottom_up(data)
        hid_probs = []
        for layer in self.upnet.layers:
            hid_probs.append(layer.probs)

        wake_deltas = []
        #Iterate over each layer excluding bottom layer
        for i in range(self.upnet.numlayers - 1):
            upper_state = hid_states[i+1]
            lower_state = hid_states[i]
            lower_activity = hid_probs[i]

            delta = dot(upper_state.transpose(), (lower_state - lower_activity))/data.shape[0]

            wake_deltas.insert(0,delta)

        return wake_deltas, hid_states[-1]

    def sleep_phase(self, data):
        '''The last step of wake-sleep and contrastive wake-sleep. Returns sleep_deltas, a list of matrices by which the
        the weights of the up net should be adjusted. Assumes DBN layers are binary stochastic layers.'''
        #Get the states and probabilities of every layer after doing a top-down pass
        hid_states = self.top_down(data)
        hid_probs = []
        for layer in self.downnet.layers:
            hid_probs.append(layer.probs)

        sleep_deltas = []
        #Iterate over each layer excluding top layer
        for i in range(self.downnet.numlayers -1):
            lower_state = hid_states[i+1]
            upper_state = hid_states[i]
            upper_activity = hid_probs[i]

            delta = dot(lower_state.transpose(), (upper_state - upper_activity))/data.shape[0]
            sleep_deltas.insert(0,delta)

        return sleep_deltas

    def wake_sleep(self, data, learning_rate):
        '''Combines wake and sleep phases'''

        downnet_deltas, top_state = self.wake_phase(data)
        upnet_deltas = self.sleep_phase(top_state) #The top state is the input for the top-down pass

        for i in range(len(downnet_deltas)):
            self.downnet.weights[i] += learning_rate*downnet_deltas[i]
        for i in range(len(upnet_deltas)):
            self.upnet.weights[i] += learning_rate*upnet_deltas[i]

class DBN(object):
    def __init__(self, bottom_layers, top_layer_rbm):
        '''Initializes a DBN consisting of an RBM on top of a BN.'''
        self.bottom_layers = bottom_layers
        self.top_layer_rbm = top_layer_rbm

    @classmethod
    def from_rbms(cls, rbms):
        '''Alternate constructor for DBN takes a list of RBM's'''
        numrbms = len(rbms)

        bottom_layers = BN.from_rbms(rbms[:numrbms-1])
        top_layer_rbm = rbms[numrbms-1]

        return cls(bottom_layers, top_layer_rbm)
        # BN(layers[:numlayers-1], btm_lyrs_up_weights, btm_lyrs_down_weights)
        # RBM.from_layers(self.bottom_layers[-1], layers[numlayers-1], top_rbm_vishid)

    def generate_data(self, startingstate, k, visdata_func=None, bn_data_func=None):
        '''To generate data from the dbn, we perform k steps of gibbs sampling given the
        state of the hiddens of the top_layer_rbm, then do a top_down pass on the bottom_layers'''
        hidstates = startingstate
        for i in range(k):
            visstates = self.top_layer_rbm.sample_vis(startingstate)
            if visdata_func is not None:
                visstates = visdata_func(visstates)
            hidstates = self.top_layer_rbm.sample_hid(visstates)
        if bn_data_func is not None:
            visstates = bn_data_func(visstates)
        activities = self.bottom_layers.top_down(visstates)
        return activities

    def contrastive_wake_sleep(self, data, K=1, learning_rate=1, rbm_data_func=None, bn_data_func=None): #Changed default learning rate, now it seems to be more useful
        '''Combines wake, CD, and sleep phases'''

        downnet_deltas, top_state = self.bottom_layers.wake_phase(data)

        #Use samples of the top of the net as the input data of the top level RBM
        #Train top level RBM using CD-k, this will adjust the weight matrix of top RBM alone
        if rbm_data_func is not None:
            top_state = rbm_data_func(top_state)
        self.top_layer_rbm.train(top_state, K, learning_rate, weightcost=0.1, dropoutrate=0)

        #Get a vis state from RBM after CD-k, use this as data for top-down pass
        top_state = self.top_layer_rbm.sample_vis(self.top_layer_rbm.sample_hid(top_state))
        if bn_data_func is not None:
            top_state = bn_data_func(top_state)
        upnet_deltas = self.bottom_layers.sleep_phase(top_state)

        for i in range(len(downnet_deltas)):
            self.bottom_layers.downnet.weights[i] += learning_rate*downnet_deltas[i]
        for i in range(len(upnet_deltas)):
            self.bottom_layers.upnet.weights[i] += learning_rate*upnet_deltas[i]
