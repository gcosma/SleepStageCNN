% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma %
% Programmed by Gulrukh Turabee at Nottingham Trent University  %
% Last revised:  2019  %
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters
% in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------



%FIND THE ASSOCIATION BETWEEN LEARNED KERNELS AND EEG EPOCH OF SLEEP STAGE 4:

%STEP#1: This Matlab script computes the Welch PSD of all sleep stage 4 Epoch from 34 out of 39 subjects in order to find the top 5 Epochs with higest peak. Remaining 5 subjects doesn't have sleep stage 4 EEG Epoch.

clear;
ss1=[];
Fs=100;
for i=1:39
    
    a = sprintf('DataTestSub_%d.mat',i)
    b = sprintf('predictionTestSub%d.mat',i);
    c= load(a);
    load(b);
    [dummy,predicted_label] = max(predY');
    index = find(predicted_label==4);
    if isempty(index)
       i=i+1;  
    Sleepstage4 = c.X_test(index,:);
    M = bsxfun(@rdivide,Sleepstage4,std(Sleepstage4));
    [pxx1,F1]= pwelch(M',[],[],[],Fs);
    idx=find(F1<3 & F1>1)
    pxx_spindle = pxx1(idx,:);
    pxx_spindle_band = sum(pxx_spindle,1);
    [idxx,temp] = sort(pxx_spindle_band , 'descend' );
    n=temp(1:5)
    ss1=[ss1;n]
    
    end
end

%As an example, plot welch PSD of sleep stage 4 for subject no.2

i=2;
a = sprintf('DataTestSub_%d.mat',i)
b = sprintf('predictionTestSub%d.mat',i);
load(a);load(b);
[dummy,predicted_label] = max(Y_pred');
index = find(predicted_label==4);
Sleepstage4 = X_test(index,:);
M = bsxfun(@rdivide,Sleepstage4,std(Sleepstage4)); % Normalize
[pxx,F]= pwelch(M',[],[],[],Fs);
idx=find(F<3 & F>1)
pxx_spindle = pxx(idx,:);
pxx_spindle_band = sum(pxx_spindle,1);
[idx,temp] = sort(pxx_spindle_band , 'descend' );
n=temp(1:5)
plot(pxx_spindle_band,'r.')
hold on;
grid;
title('Power Spectrum Estimate of Subject No.2 Epochs')
xlabel('Epoch')
ylabel('Power Spectral Density [V**2/Hz]')

% STEP#4: Plot Power Spectrum (PSD) using Welch PSD of Top 5 Epoch from Subject No.2 in order to associate learned kernels with EEG Patterns.

figure,loglog(F,pxx(:,117))
hold on;
loglog(F,pxx(:,88))
loglog(F,pxx(:,96))
loglog(F,pxx(:,24))
loglog(F,pxx(:,4))
title('Welch Power Spectrum Density Estimate')
ylabel('Power Spectral Density [V**2/Hz]')
xlabel ('Frequency [Hz]')


