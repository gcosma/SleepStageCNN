# -*- coding: utf-8 -*-
"""
% Code adapted from: Supratak, A., Dong, H., Wu, C. and Guo, Y., 2017. DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(11), pp.1998-2008.
Link to download : https://github.com/kedarps/DeepSleepNet
"""

"""
Test DeepSleepNet on testing data
"""

from __future__ import print_function

from glob import glob

import os
import numpy as np
import deepSleepNet as dsn
import readMat as mat
import scipy.io as io
from keras.models import Model
from keras import backend as K
from scipy.io import loadmat
from scipy.io import savemat

# path where files live
dataDir = os.path.join(os.getcwd(), 'data')
nnDir = os.path.join(os.getcwd(), 'preTrainResults')
saveDir = os.path.join(os.getcwd(), 'results','supPreTrain+Test')
all_Y_pred = [];
n_feats = 3000
n_classes = 4

# initialize deep sleep
preTrainNN = dsn.preTrainingNet(n_feats, n_classes)

"""Copyright (c) 2019, Gulrukh Turabee et. al. All rights reserved."""
#save the model as HDF5 file
preTrainNN.save('my_model.h5')
files = os.listdir(dataDir)
accuracies = []

for f in files:
    file = os.path.join(dataDir, f)
    
    (X_test, Y_test), _ = mat.getTestingData(file)
    testIdx = (f.split('.')[0]).split('_')[1]
    print('Testing on sub {}...\n'.format(testIdx))
    
    # reshape to keep input to NN consistent
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
     
    # set weights from pre-trained network
    psrName = os.path.join(nnDir, 'supervisePreTrainNet_TestSub'+testIdx+'.h5')
    preTrainNN.load_weights(psrName, by_name=True)
   
    Y_pred = preTrainNN.predict(X_test, verbose=1)
    
    """Copyright (c) 2019, Gulrukh Turabee et. al. All rights reserved."""

    preTrainNN.save('my_model.h5')
    savemat(os.path.join(saveDir, 'predictionTestSub'+testIdx+'.mat'),dict([('Y_pred', Y_pred),('Y_true',Y_test)]));

    # now with saved network run testing data
    scores = preTrainNN.evaluate(X_test, Y_test, verbose=1)
    accuracies.append([int(testIdx), scores[1]*100])
    print('Accuracy = {} %...\n'.format(scores[1]*100))
np.savetxt('preTrainTestingAccuracies.csv', np.array(accuracies), delimiter=';')





