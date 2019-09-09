% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma %
% Programmed by Gulrukh Turabee at Nottingham Trent University              %
% Last revised:  2019  %
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters
% in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------


% STEP#2: This Matlab script finds the unique kernels of those sleepstage 4 epoch with higest peaks exacted from welch PSD ( step 1)

nSubject = 39;
for i=1:6
    running_index = 1:nSubject;
    tmp_index = find(running_index ~= 7 &...
        running_index ~= 10 &...
        running_index ~= 15 &...
        running_index ~= 19 &...
        running_index ~= 20);
end

for i=1:34
    j= tmp_index(1,i);
    a=sprintf('activations%d.mat',j);
    load(a);
    buf = squeeze(sum(all_activations,2));
    index = ss1(i,:);
    activation_map = buf(index,:);
    activation_ss1 = squeeze([activation_ss1;activation_map])
end



all_idx1=[];
for k=1:170
    [sortx,idx] = sort(activation_ss1(k,:),'descend');
    all_idx1 = [all_idx1;idx];
end

u=unique(all_idx1(:,1))
