% Example of using a datastore, see 

clc;

rootdir = 'SunsetImages/';
subdir = [rootdir 'train'];

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% Make datastores for the validation and testing sets similarly.

fprintf('Read images into datastores\n');

xTrain = imageDatastoreReader(trainImages);
yTrain = trainImages.Labels;

% idx = find(yTrain == 'nonsunset');
% NEWyTrain(idx) = -1;
% idx = find(yTrain == 'sunset');
% NEWyTrain(idx) = 1;

%% Train and evaluate an SVM

