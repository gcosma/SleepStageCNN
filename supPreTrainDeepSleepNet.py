# -*- coding: utf-8 -*-
"""
% Code adapted from: Supratak, A., Dong, H., Wu, C. and Guo, Y., 2017. DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(11), pp.1998-2008.
Link to download : https://github.com/kedarps/DeepSleepNet
"""

"""
Trains neural network on oversampled training data
"""
import tensorflow as tf
import os
import numpy as np
import readMat as mat
import deepSleepNet as dsn
import scipy.io as io
import matplotlib.pyplot as plt
from time import time


all_weights = [];

# path where files live
dataDir = os.path.join(os.getcwd(), 'data')
files = os.listdir(dataDir)
saveDir = os.path.join(os.getcwd(),'PreTrainResults')

for f in files:
    file = os.path.join(dataDir, f)
    print(file);
    (X_train, Y_train) = mat.getTrainingData(file)
    
    (X_train_ovs, Y_train_ovs) = mat.oversample_minority_class(X_train, Y_train)
    testIdx = (f.split('.')[0]).split('_')[1]
    
    # reshape to keep input to NN consistent
    X_train_ovs = np.reshape(X_train_ovs, (X_train_ovs.shape[0], X_train_ovs.shape[1], 1))
    
    (n_samples, n_feats, _) = X_train_ovs.shape
    (_, n_classes) = Y_train_ovs.shape
    
    # pre-training phase
    preTrain = dsn.preTrainingNet(n_feats, n_classes)
    
    # train this network on oversampled dataset
    preTrain.fit(X_train_ovs, Y_train_ovs, epochs=75,batch_size=100)  
    
    # save neural network weights so that we can use them while testing    
    preTrain.summary();
    preTrain.save_weights(os.path.join(saveDir,'supervisePreTrainNet_TestSub'+ testIdx +'.h5'));    
    
    """Copyright (c) 2019, Gulrukh Turabee et. al. All rights reserved."""
    #get and save weights
    for layer in preTrain.layers:
        weights = layer.get_weights() # list of numpy arrays
        all_weights.append(weights);
    #save weights in Matlab (.mat) files.
    io.savemat(os.path.join(saveDir,'SupervisePreTrainNet_TestSub'+ testIdx +'.mat'), dict([('weights', all_weights)]))

