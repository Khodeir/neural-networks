from network import *
from layer import *
from metrics import error
from backprop import backprop, train

class AutoEncoder(object):
    '''An Autoencoder is made of an encoder and decoder, specified by the layer sizes.'''
    def __init__(self, e_layers, d_layers, e_weights=None, d_weights=None):
        '''Initialise Autoencoder as 2 networks, one encoding and one decoding'''
        self.encoder = NeuralNet(e_layers, e_weights)
        self.decoder = NeuralNet(d_layers, d_weights)

    @classmethod
    def from_architecture(cls, architecture, training_inputs):
        '''Creates an Autoencoder for the training_inputs by stacking shallow autoencoders. architecture is a list of layers for autoencoder.
        Also returns the layer by layer training errors in a list.'''
        #train_errors = []
        e_layers, e_weights, d_layers, d_weights = [],[],[],[]
        #Initial values for first layer
        next_layer = architecture[0]
        next_input = training_inputs[:]
        e_layers.append(next_layer) #Add first layer of encoder

        for i in range(len(architecture)-1): #Iterate through architecture, generating a new net and training it.
            third_layer = LogisticLayer(next_layer.size)
            current = NeuralNet([next_layer, architecture[i+1], third_layer])
            print 'Training ', current.layers[0].size, ' - ', current.layers[1].size, ' encoder'
            error = train_ae(current, next_input)
            #train_errors.append(error)

            next_layer = current.layers[1] #Get middle layer of this shallow autoencoder
            next_input = current.forward_pass(next_input)[1][:] #Encode data to get next_input

            e_layers.append(next_layer) #Add this to encoder
            e_weights.append(current.weights[0])
            d_layers.insert(0, current.layers[2])#Add layer to decoder
            d_weights.insert(0, current.weights[1])

        d_layers.insert(0, current.layers[1]) #Final decoder layer added

        return cls(e_layers, d_layers, e_weights, d_weights) #, train_errors

    def encode(self, data):
        '''Encodes data and returns it.'''
        return self.encoder.forward_pass(data)[-1]

    def decode(self, data):
        '''Decode data and returns it.'''
        return self.decoder.forward_pass(data,1)[-1] #Need to verify this

    def fine_tune(self, training_inputs):
        '''Fine Tune a deep autoencoder using backprop on the whole net after it's been pre-trained'''
        fine_layers, fine_weights = [],[]

        for layer in self.encoder.layers:
            fine_layers.append(layer)
        for matrix in self.encoder.weights:
            fine_weights.append(matrix)

        for i in range(self.decoder.numlayers -1):
            fine_layers.append(self.decoder.layers[i+1])
        for matrix in self.decoder.weights:
            fine_weights.append(matrix)

        fine_net = NeuralNet(fine_layers, fine_weights)

        err = train_ae(fine_net, training_inputs, epochs=5, learning_rate=0.05, decay_rate=0)
        encode_weights, decode_weights = [],[]
        for i in range(0,(len(fine_net.weights)/2)):
            encode_weights.append(fine_net.weights[i])
        for i in range((len(fine_net.weights)/2),len(fine_net.weights)):
            decode_weights.append(fine_net.weights[i])

        self.encoder.set_weights(encode_weights)
        self.decoder.set_weights(decode_weights)

    def insert_layer(self, index, layer):
        '''Adds an intermediate encoding/decoding layer to the autoencoder at the given index'''

def train_ae(ae, training_inputs, epochs=5, learning_rate=0.1, batch_size=100, noise=0, decay_rate=0, get_error=False):
    '''Trains an assembled autoencoder on data.'''
    training_inputs = dropout(training_inputs, noise) #Add noise to input if you want
    input_batches = [training_inputs[i:i+batch_size] for i in range(0, 60000, batch_size)] #Split to mini-batches
    errors = []

    for epoch in range(epochs):
        print 'Epoch Number: ', epoch
        for i in range(len(input_batches)):
            train(ae, input_batches[i], input_batches[i], learning_rate, decay_rate)
            if i%100 == 0 and get_error:
                e = error(ae, input_batches[i], input_batches[i], decay_rate)
                print e
                errors.append(e)
    if get_error:
        return errors
