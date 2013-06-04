from dbn import BN, DBN
from rbm import RBM
from rbmstack import RBMStack
from scipy.io import loadmat

training_data = loadmat('data.mat')['data']['training'][0][0][0][0]
validation_data = loadmat('data.mat')['data']['validation'][0][0][0][0]

# There are 1000 training data cases
training_input = training_data[0].transpose()
training_target = training_data[1].transpose()

architecture = [RBM(256, 300), RBM(300, 300), RBM(300, 300)]

stack = RBMStack(training_input, architecture)

digitdbn = DBN.from_rbms(architecture)

#just test that the layers are what they're supposed to be
assert digitdbn.bottom_layers.upnet.numlayers == 3

assert digitdbn.bottom_layers.upnet.layers[0] is architecture[0].get_vislayer()
assert digitdbn.bottom_layers.upnet.layers[1] is architecture[0].get_hidlayer()
assert digitdbn.bottom_layers.upnet.layers[2] is architecture[1].get_hidlayer()

assert digitdbn.bottom_layers.downnet.layers[0] is architecture[1].get_hidlayer()
assert digitdbn.bottom_layers.downnet.layers[1] is architecture[0].get_hidlayer()
assert digitdbn.bottom_layers.downnet.layers[2] is architecture[0].get_vislayer()

assert digitdbn.top_layer_rbm.get_vislayer() is architecture[1].get_hidlayer()
assert digitdbn.top_layer_rbm.get_hidlayer() is architecture[2].get_hidlayer()
#all good

#to train the individiual rbms, use:
    #stack.train(self, macindex, K=1, epochs=100, learning_rate=0.1, weightcost=0.1, dropoutrate=0)
#after you're done training the rbms, to untie the recognition weights from the generative ones, run:
    #digitdbn.bottom_layers.__untie_weights__()
