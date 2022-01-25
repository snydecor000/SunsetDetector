%% Alexnet Feature Extraction with SVM
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');

%% Extract Training Data
clc; clear all;

rootdir = 'SunsetImages/';
subdir = [rootdir 'train'];

fprintf('Read images into datastores\n');

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

net = alexnet;
inputSize = net.Layers(1).InputSize;
resizedTrainImgs = augmentedImageDatastore(inputSize(1:2),trainImages);
layer = 'fc7';

xCNNTrain = activations(net,resizedTrainImgs,layer,'OutputAs','rows');
yCNNTrain = trainImages.Labels;

% idx = find(yCNNTrain == 'nonsunset');
% NEWyCNNTrain(idx) = -1;
% idx = find(yCNNTrain == 'sunset');
% NEWyCNNTrain(idx) = 1;
% yCNNTrain = NEWyCNNTrain;
% save('yCNNTrain.mat')
% save('xCNNTrain.mat')

%% Extract Validation Data
clc; clear all;

rootdir = 'SunsetImages/';
subdir = [rootdir 'validate'];

fprintf('Read images into datastores\n');

validateImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

net = alexnet;
inputSize = net.Layers(1).InputSize;
resizedValidateImgs = augmentedImageDatastore(inputSize(1:2),validateImages);
layer = 'fc7';

xCNNValidate = activations(net,resizedValidateImgs,layer,'OutputAs','rows');
%%
yCNNValidate = validateImages.Labels;

idx = find(yCNNValidate == 'nonsunset');
NEWyCNNValidate(idx) = -1;
idx = find(yCNNValidate == 'sunset');
NEWyCNNValidate(idx) = 1;
yCNNValidate = NEWyCNNValidate;
save('yCNNValidate.mat')
save('xCNNValidate.mat')