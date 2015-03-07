neural-networks
===============

A suite of neural-network architectures, training algorithms and useful utility functions written in python.

A short example usecase (data not actually provided):

    from network import NeuralNet
    from backprop import train
    from layer import LogisticLayer, LinearLayer, SoftMaxLayer
    from metrics import error
    
    #to load matlab format data
    from scipy.io import loadmat
    
    mnist = loadmat('MNIST60k.mat')
    
    training_data = mnist['train']
    validation_data = mnist['valid']
    
    model = NeuralNet([LinearLayer(784), LogisticLayer(1000), SoftMaxLayer(10)])
    
    epochs = 1000
    
    #Train the model using backprop
    for epoch in range(epochs):
        
        #do a single iteration of backprop
        train(model, training_data['input'], training_data['target'])
        
        #evaluate on validation set
        valid_error = error(model, validation_data['input'], validation_data['target'])
        print 'EPOCH %d VALID CROSS ENTROPY %.5e' % valid_error
     
