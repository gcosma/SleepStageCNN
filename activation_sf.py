# -*- coding: utf-8 -*-
"""
% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma 
% Programmed by Gulrukh Turabee at Nottingham Trent University
% Last revised:  2019
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
-----------------------------------------------------------------
"""

"""
This code implements the 1st CNN layer of model again using the sequencial model of keras and the saved weights are 
loaded from previous model and are set to this model to extract the activations which are then saved as MATLAB (.mat) files. 
"""

from keras.models import Model
from keras.layers import Conv1D

import tensorflow as tf
import os
import numpy as np
import readMat as mat
import deepSleepNet as dsn
import scipy.io as io
import matplotlib.pyplot as plt
from time import time
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense,MaxPooling1D
from keras.models import load_model
from scipy.io import loadmat
from scipy.io import savemat


Fs = 100
n_feats = 3000
n_classes = 4
dataDir = os.path.join(os.getcwd(), 'data')
files = os.listdir(dataDir)
saveDir = os.path.join(os.getcwd(),'activations','activation_sf')

#make a new model with 1st layer
model = Sequential()
model.add(Conv1D(64,input_shape=(n_feats,1),kernel_size=int(Fs/2),strides=int(Fs/16),  padding='same',activation='relu', name='fConv1'))

# load the saved model in order to extract activations
previous_model= load_model('my_model.h5')

for f in files:
    file = os.path.join(dataDir, f)
    (X_test, Y_test), _ = mat.getTestingData(file)
    testIdx = (f.split('.')[0]).split('_')[1]
    print('Testing on sub {}...\n'.format(testIdx))
    
    # reshape to keep input to NN consistent
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    weights = previous_model.layers[1].get_weights()
    model.layers[0].set_weights(weights)
    np.std(X_test)
    activations = model.predict(X_test,verbose=1)
    print(activations.shape);
    model.summary();
    io.savemat(os.path.join(saveDir,'activations'+ testIdx +'.mat'), {'all_activations':activations});
