from network import NeuralNet


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
            self.downnet.weights[i] = self.upnet.weights[i].copy()

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
