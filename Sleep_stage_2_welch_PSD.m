% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma %
% Programmed by Gulrukh Turabee at Nottingham Trent University %
% Last revised:  2019  %
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters
% in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------


%FIND THE ASSOCIATION BETWEEN LEARNED KERNELS AND EEG EPOCH OF SLEEP STAGE 2:

%STEP#1: This Matlab script computes the Welch PSD of all sleep stage 2 Epoch from 34 subjects out of 39 subjects in order to find the top 5 Epochs with higest peak. 
%Remaining 5 subjects doesn't have sleep stage 2 EEG Epoch.

clear;
Fs=100;
ss=[];

for i=1:39
    
    a = sprintf('DataTestSub_%d.mat',i)
    b = sprintf('predictionTestSub%d.mat',i);
    load(a);load(b);
    [dummy,predicted_label] = max(Y_pred');
    index = find(predicted_label==2);
    if isempty(index)
       i=i+1;  
    Sleepstage2 = X_test(index,:);
    M = bsxfun(@rdivide,Sleepstage2,std(Sleepstage2)); %Normalize
    [pxx,F]= pwelch(M',[],[],[],Fs);
    idx=find(F<15 & F>13)
    pxx_spindle = pxx(idx,:);
    pxx_spindle_band = sum(pxx_spindle,1);
    [idx,temp] = sort(pxx_spindle_band , 'descend' ); % sort in desending order
    n=temp(1:5)
    ss=[ss;n]
    
    end
end


% STEP#2:VALIDATION OF WELCH PSD:Plot Power Spectrum (PSD) using Welch PSD of Top 5 Epoch from Subject No.2 in order to associate learned kernels with EEG Pattern.

Fs=100;
i=2;
a = sprintf('DataTestSub_%d.mat',i)
b = sprintf('predictionTestSub%d.mat',i);
load(a);load(b);
[dummy,predicted_label] = max(Y_pred');
index = find(predicted_label==2);
Sleepstage2 = X_test(index,:);
M = bsxfun(@rdivide,Sleepstage2,std(Sleepstage2)); % Normalize
[pxx,F]= pwelch(M',[],[],[],Fs);
idx=find(F<15 & F>13)
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


figure,loglog(F,pxx(:,n))
    %OR% 
figure,loglog(F,pxx(:,15))
hold on;
loglog(F,pxx(:,282))
loglog(F,pxx(:,353))
loglog(F,pxx(:,63))
loglog(F,pxx(:,9))
title('Welch Power spectrum Density Estimate')
ylabel('Power Spectral Density [V**2/Hz]')
xlabel ('Frequency [Hz]')

% Top 2 selected Epoch EEG pattern plotted

figure;plot((1:3000)/100,Sleepstage2(15,:));grid;
title('Epoch No.15 from 2nd Subject');
figure;plot((1:3000)/100,Sleepstage2(282,:));grid;
title('Epoch No.282 from 2nd Subject');

