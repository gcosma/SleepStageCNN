# -*- coding: utf-8 -*-

"""
% Code adapted from: Supratak, A., Dong, H., Wu, C. and Guo, Y., 2017. DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(11), pp.1998-2008.
Link to download : https://github.com/kedarps/DeepSleepNet
"""

"""
Module for importing Training and Testing Data from .mat files in Matlab. 
Ensure to save .mat files using the '-v7.3' flag since this uses h5py package
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import hdf5storage


def getTrainingData(file_name, one_hot_encode=True, min_max_scale=False):
    print('Reading mat file...')
    # read -v7.3 .mat files using h5py, make sure to save them using this flag in matlab
    # this will give array of arrays
    a = hdf5storage.loadmat(file_name)
    X_train = a['X_train'] 
    l = len(X_train)
    X_train_seq = []
    for k in range(l):
        buf = X_train[k];
        buf = np.asarray(buf[0])
        X_train_seq.append(buf)
    X_train = np.concatenate(X_train_seq)  
    Y_train = a['Y_train'] 
    l = len(Y_train)
    Y_train_seq = []
    for k in range(l):
        buf = Y_train[k];
        buf = np.asarray(buf[0])
        Y_train_seq.append(buf)
    Y_train = np.concatenate(Y_train_seq)  
     
    print('Consolidating classes N3 and N4...')
    # get rid of class 3, merge it with class 4
    Y_train[Y_train == 3] = 4
    Y_train[Y_train == 6] = 7
    Y_train[Y_train == 7] = 0
    
    print('removing sleep stage wake == 0'); 
    l = len(Y_train)
    Y_train_new = []
    X_train_new = []
    index= 0;
    for k in range(l):
        buf = Y_train[k];
        buf1 = X_train[k];
        if buf != index:
           Y_train_new.append(buf)
           X_train_new.append(buf1)
    Y_train_new = np.asarray(Y_train_new);
    X_train_new = np.asarray(X_train_new);
  
    
    
    if min_max_scale:
        print('Standardizing dataset...')
        X_train_new = scale_data(X_train_new)
            
          
    if one_hot_encode:
        print('Running one-hot-encoding...')
        # one hot encoding for output labels 
        Y_train_new = one_hot_encode_data(Y_train_new.reshape(-1, 1))
        
    
    
    return (X_train_new, Y_train_new)

def appendZeros(inData, num_zeros=10):
    outData = np.append(inData, np.zeros((inData.shape[0], num_zeros)), axis=1)
    
    return outData
    

def getTestingData(file_name, one_hot_encode=True, min_max_scale=False):
    
    print('Reading mat file...')
    a = hdf5storage.loadmat(file_name)
    X_test = a['X_test'] 
    
    l = len(X_test)
    X_test_seq = [];
    for k in range(l):
        buf = X_test[k];
        X_test_seq.append(buf)
    X_test_seq = np.asarray(X_test_seq);   
    Y_test = a['Y_test'] 
    l = len(Y_test)
    Y_test_seq = []
    for k in range(l):
        buf = Y_test[k];
        Y_test_seq.append(buf)
    Y_test_seq = np.asarray(Y_test_seq);
   
    
    # testing index, it is corrected for python's zero-indexing
    testIdx = a['TestSubIdx']- 1
        
    print('Consolidating classes N3 and N4...')
    # get rid of class 3, merge it with class 4
    Y_test_seq[Y_test_seq == 3] = 4
    Y_test_seq[Y_test_seq == 6] = 7
    Y_test_seq[Y_test_seq == 7] = 0
    
    print('removing sleep stage== 0'); 
    l = len(Y_test)
    Y_test_new = []
    X_test_new = []
    index= 0;
    for k in range(l):
        buf = Y_test_seq[k];
        buf1 = X_test_seq[k];
        if buf != index:
           Y_test_new.append(buf)
           X_test_new.append(buf1)
    Y_test_new = np.asarray(Y_test_new);
    X_test_new = np.asarray(X_test_new);
    
    if min_max_scale:
        print('Standardizing dataset...')
        X_test_new = scale_data(X_test_new)
    
    if one_hot_encode:
        print('Running one-hot-encoding...')
        # one hot encoding for output labels
        Y_test_new = one_hot_encode_data(Y_test_new.reshape(-1,1))
        
    return (X_test_new, Y_test_new), testIdx
    
def scale_data(inData):
    scaler = MinMaxScaler()
    outData = scaler.fit_transform(inData)
   
    
    return outData

def label_encode_data(inData):
    ohe = LabelEncoder()
    outData1 = ohe.fit_transform(inData)
    
    return outData1

def one_hot_encode_data(inData,categorical_features='auto', categories='auto'):
    ohe = OneHotEncoder()
    outData = ohe.fit_transform(inData).toarray()
    
    return outData

def oversample_minority_class(X, Y):
    print('Oversampling minority classes based on number of instances in Class 5...')
    
    # this does one hot decoding, but since we removed class 3, we only have 5 classes
    # and now after argmax, 3 corresponds to class 5
    Y = np.argmax(Y, axis=1)
    #print(Y);
    
    # we will oversample number of samples in class 5, which corresponds to 3 in 
    # OHD Y
         
    majClass = 2
    nObsPerClass = sum( Y == majClass )
    nFeats = X.shape[1]
    
    # initialize empty arrays with (num_5 * 6) observations
    X_out = np.empty( (nObsPerClass*4, nFeats) )
    Y_out = np.empty( (nObsPerClass*4,) )
    
    ptr = 0
    for l in np.unique(Y):
        X_here = X [ Y == l,: ]
        
       
        if l == majClass:
            X_here_ovs = X_here
        else:
            X_here_ovs = resample(X_here, n_samples=nObsPerClass)
        
        X_out[ ptr:ptr+nObsPerClass,: ] = X_here_ovs
        Y_out[ ptr:ptr+nObsPerClass ] = l * np.ones((nObsPerClass,))
        
        ptr += nObsPerClass
    
    Y_out = one_hot_encode_data(Y_out.reshape(-1, 1))
   
    return (X_out, Y_out)
    