% Interpreting the Filters in the First Layer of a Convolutional Neural Network for Sleep Stage Classification by Gulrukh Turabee , Yuan Shen and Georgina Cosma %
% Programmed by Gulrukh Turabee at Nottingham Trent University %
% Last revised:  2019  %
% Reference: Turabee G., Shen Y., Cosma G. (2020) Interpreting the Filters
% in the First Layer of a Convolutional Neural Network for Sleep Stage Classification. In: Ju Z., Yang L., Yang C., Gegov A., Zhou D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham.
% Copyright (c) 2019, Gulrukh Turabee , Yuan Shen and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------



% This code compiles all the .edf files(EDF Recordings) into one .mat file.Make two cell arrays to save 39 sleepStageScore and SleepRecoring

sleepStageScore = cell(39, 1);
sleepRecording = cell(39, 2);

%Run the code below 39 times with each subject (predecessor files) to save the data.Note that 'SC4001E0' is a subject-specifying label for PSG file and  'SC4001EC' for Hypnogram file;
% For extracting other subjects' EEG data, please update these labels accordingly.

%subject 1

fname='SC4001E0-PSG.edf'
fname1='SC4001EC-Hypnogram.edf';
[dummy, record] = edfread(fname);
[dummy, record1] = edfread(fname1);
sleepRecording{1,1} = record(1,:);
sleepRecording{1,2} = record(2,:);
sleepStageScore{1} = record1(1,:);
allData.sleepStageScore = sleepStageScore;
allData.sleepRecording = sleepRecording;

...........

%subject =39

fname='SC4192E0-PSG.edf'
fname1='SC4192EV-Hypnogram.edf';
[dummy, record] = edfread(fname);
[dummy, record1] = edfread(fname1);
sleepRecording{39,1} = record(1,:);
sleepRecording{39,2} = record(2,:);
sleepStageScore{39} = record1(1,:);
allData.sleepStageScore = sleepStageScore;
allData.sleepRecording = sleepRecording;

%After running the above code for 39 times, run the following code line to save the allData as structure in another Matlab (.mat) File.
save('DeepSleepNetData.mat','-v7.3','-struct','allData');
