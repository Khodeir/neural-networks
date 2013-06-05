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
        return self.upnet.forward_pass(data, 1)

    def top_down(self, data):
        return self.downnet.forward_pass(data, 1)


    def __untie_weights__(self):
        '''This is an ugly step, and is only necessary when the db is initialized from RBMs.
        It unties the recognition weights from the generative ones.'''
        numweights = self.numlayers - 1
        for i in range(numweights):
            self.downnet.weights[i] = self.downnet.weights[i].copy()

    def wake_phase(self, data):
        '''The first step of wake-sleep and contrastive wake-sleep. Returns wake_deltas, a list of matrices by which the
        the weights of the down net should be adjusted. Also returns hidden states of top layer.'''
        #Get the activities (probabilities) of every layer after doing a bottom-up pass
        #hid_probs is a list of lists, each one containing the activations for each layer, starting with lowest layer. 
        hid_probs = self.bottom_up(data)
        #Get binary states for these layers stochastically.
        hid_states = []
        for layer in hid_probs:
            hid_states.append(sample_binary_stochastic(layer))

        wake_deltas = []

        for i in range(upnet.numlayers -1) #Iterate over each layer excluding bottom layer
            upper_state = hid_states[i+1]
            lower_state = hid_states[i]
            lower_activity = hid_probs[i]

            delta = dot(upper_state.transpose(), (lower_state - lower_activity))/data.shape(0)

            wake_deltas.insert(0,delta)

        return wake_deltas, hid_states[-1]

    def sleep_phase(self, data):
        '''The last step of wake-sleep and contrastive wake-sleep. Returns sleep_deltas, a list of matrices by which the
        the weights of the up net should be adjusted. '''
        #Get the activities (probabilities) of every layer after doing a top-down pass
        #hid_probs is a list of lists, each one containing the activations for each layer, starting with top layer. 
        hid_probs = self.top_down(data)
        #Get binary states for these layers stochastically.
        hid_states = []
        for layer in hid_probs:
            hid_states.append(sample_binary_stochastic(layer))

        sleep_deltas = []

        #Iterate over each layer excluding top layer
        for i in range(self.downnet.numlayers -1):
            lower_state = hid_states[i+1]
            upper_state = hid_states[i]
            upper_activity = hid_probs[i]

            delta = dot(lower_state.transpose(), (upper_state - upper_activity))/data.shape(0)
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
        assert top_layer_rbm.get_vislayer() is bottom_layers.upnet.layers[-1]
        assert top_layer_rbm.get_vislayer() is bottom_layers.downnet.layers[0]

    @classmethod
    def from_rbms(cls, rbms):
        '''Alternate constructor for DBN takes a list of RBM's'''
        numrbms = len(rbms)

        bottom_layers = BN.from_rbms(rbms[:numrbms-1])
        top_layer_rbm = rbms[numrbms-1]
        top_layer_rbm.layers[0] = rbms[numrbms-2].get_hidlayer()

        return cls(bottom_layers, top_layer_rbm)
        # BN(layers[:numlayers-1], btm_lyrs_up_weights, btm_lyrs_down_weights)
        # RBM.from_layers(self.bottom_layers[-1], layers[numlayers-1], top_rbm_vishid)

    def generate_data(self, startingstate, k):
        '''To generate data from the dbn, we perform k steps of gibbs sampling given the
        state of the hiddens of the top_layer_rbm, then do a top_down pass on the bottom_layers'''
        print 'starting gibbs with', startingstate.shape
        visprobs_top_rbm = self.top_layer_rbm.gibbs_given_h(startingstate, k)[0]
        activities = self.bottom_layers.top_down(visprobs_top_rbm)
        return activities

    def contrastive_wake_sleep(self, data, K=1, learning_rate=0.1):
        '''Combines wake, CD, and sleep phases'''
        
        downnet_deltas, top_state = self.wake_phase(data)

        #Use samples of the top of the net as the input data of the top level RBM
        #Train top level RBM using CD-k, this will adjust the weight matrix of top RBM alone
        top_layer_rbm.train(top_state, K, epochs=1, learning_rate, weightcost=0.1, dropoutrate=0))

        top_activities = top_layer_rbm.get_vislayer().activities
        top_state = sample_binary_stochastic(top_activities)
        #Get a vis state from RBM after CD-k, use this as data for top-down pass
        upnet_deltas = self.sleep_phase(top_state)

        for i in range(len(downnet_deltas)):
            self.bottom_layers.downnet.weights[i] += learning_rate*downnet_deltas[i]
        for i in range(len(upnet_deltas)):
            self.bottom_layers.upnet.weights[i] += learning_rate*upnet_deltas[i]
