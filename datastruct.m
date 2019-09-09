% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma %
% Programmed by Gulrukh Turabee at Nottingham Trent University  %
% Last revised:  2019  %
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters
% in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------



% This code takes input of .mat file containing Sleep stage score along with sleep Recordings.Since there are 39 subjects, using leave-one-subject out cross-validation, implementing this code will make 39 .mat files with each subject 
%as testing data with the filename as DataTestSub_{subjectIndex}.mat. The training data is stored as cell arrays, X_train and Y_train, where each cell contains a matrix. 
%The testing data, X_test and Y_test is stored as matrices.

a=load('DeepSleepNetData.mat');
b= a.sleepRecording;
c =a.sleepStageScore;
nSubject = size(b,1);
lEpoch = 3000;

for subjectIndex = 1:nSubject
    
    eegChannel = b{subjectIndex,1};
    nEpoch = floor(length(eegChannel)/lEpoch);
    eegEpochs = reshape(eegChannel,lEpoch,nEpoch);eegEpochs = eegEpochs';
    X_test = eegEpochs;
    data.X_test = X_test;
    
    SleepScore = c{subjectIndex,1};
    buf1 = SleepScore(1:(end-1),2); buf1 = buf1/30;
    buf2 = SleepScore(1:(end-1),3); 
    
    sleepStage = [];
    l = length(buf1);
    for k = 1:l
        n = buf1(k);
        buf3 = ones(n,1) * buf2(k);
        sleepStage = [sleepStage; buf3];
    end
    Y_test = sleepStage;
    data.Y_test = Y_test;
    
    running_index = 1:nSubject;
    tmp_index = find(running_index ~= subjectIndex &...
        running_index ~= 4 &...
        running_index ~= 25);
    n_tmp_index = length(tmp_index);
    running_index = running_index(tmp_index);
    X_train = cell(n_tmp_index, 1);
    Y_train = cell(n_tmp_index, 1);
    index = 1;
    for k = 1:n_tmp_index
        index = running_index(k);
        eegChannel = b{index,1};
        nEpoch = floor(length(eegChannel)/lEpoch);
        eegEpochs = reshape(eegChannel,lEpoch,nEpoch);eegEpochs = eegEpochs';
        X_train{k,1} = eegEpochs;
        
        SleepScore = c{index,1};
        buf1 = SleepScore(1:(end-1),2); buf1 = buf1/30;
        buf2 = SleepScore(1:(end-1),3);
        
        sleepStage = [];
        l = length(buf1);
        for kk = 1:l
            n = buf1(kk);
            buf3 = ones(n,1) * buf2(kk);
            sleepStage = [sleepStage; buf3];
        end
        Y_train{k,1} = sleepStage;
        data.X_train = X_train;
        data.Y_train = Y_train;
        
    end
    data.TestSubIdx = subjectIndex;
    save(['DataTestSub_' num2str(subjectIndex) '.mat'],'-struct','data','-v7.3');

end 
%--------------------------------------------------------------------------------------------
    