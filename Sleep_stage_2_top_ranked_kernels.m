% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma %
% Programmed by Gulrukh Turabee at Nottingham Trent University  %
% Last revised:  2019  %
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters
% in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------


% STEP#3: This Matlab script finds the top ranked kernels of those sleepstage2 epoch with higest peaks extracted from welch PSD ( STEP 1)

nSubject = 39;
for subjectIndex =1:nSubject
running_index = 1:nSubject;
    tmp_index = find(running_index ~= 7 &...
        running_index ~= 15 &...
        running_index ~= 19 &...
        running_index ~= 20 &...
        running_index ~= 25);
   
end


activation_ss=[];
for i=1:34
    j= tmp_index(1,i);
    a=sprintf('activations%d.mat',j);
    load(a);
    buf = squeeze(sum(all_activations,2));
    index = ss(i,:);
    activation_map = buf(index,:);
    activation_ss = squeeze([activation_ss;activation_map])
end

% sort all unique kernels in desending order
all_idx=[];
for k=1:170
    [sortx,idx] = sort(activation_ss(k,:),'descend');
    all_idx = [all_idx;idx];
end

%save them in a variable for next step(STEP 4)
h=unique(all_idx(:,1))

% The occurrence frequencies of the top 9 ranked kernels. Here y is the
% lenght of each Kernel.
y= [84,52,10,5,4,4,4,4,2,1]
figure,bar(y)
title('9 Top Ranked Kernels')
ylabel('Length(No.of times')
xlabel('Kernels')
