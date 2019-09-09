% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma %
% Programmed by Gulrukh Turabee at Nottingham Trent University  %
% Last revised:  2019  %
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters
% in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------

% STEP#4: Plot Power Spectrum (PSD) using Yule-Walker Method of CNN Learned Kernels.

load('supervisePreTrainNet_TestSub2.mat')
w1 = all_weights{2};
w1=w1{1}
w1=squeeze(w1)
w11= w1(:,h)
M = bsxfun(@rdivide,w11,std(w11));
[pxx391,F1] = pyulear(M,6,1:25,Fs)
[dummy, st] = max(pxx391);
index1=find(st>12 & st<=15);
figure,plot(pxx391(:,index1)); 
title('Yule-Walker Power Spectral Density Estimate')
xlabel('Frequency [Hz]')
ylabel('Power/frequency [dB/Hz]');


