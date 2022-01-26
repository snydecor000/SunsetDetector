%% Transfer Learning on training and validation set
% Load training data
rootdir = 'SunsetImages/';
subdir = [rootdir 'train'];

fprintf('Read images into datastores\n');

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% Load validation data
rootdir = 'SunsetImages/';
subdir = [rootdir 'validate'];

fprintf('Read images into datastores\n');

validateImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% Load pretrained network
net = alexnet;
analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;

% Replace layers confifured for 1000 classes (last 3)
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainImages.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Train network
resizedTrainImgs = augmentedImageDatastore(inputSize(1:2),trainImages);
resizedValidateImgs = augmentedImageDatastore(inputSize(1:2),validateImages);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',resizedValidateImgs, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(resizedTrainImgs,layers,options);
%% Use transfered network on test images
rootdir = 'SunsetImages/';
subdir = [rootdir 'test'];

fprintf('Read images into datastores\n');

testImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

resizedTestImgs = augmentedImageDatastore(inputSize(1:2),testImages);

[testPredict, score] = classify(netTransfer, resizedTestImgs);
yActual = testImages.Labels;
accuracy_mean = mean(testPredict == yActual);

yTest = yTest';
testPredict = testPredict';

idx = find(yTest == 'nonsunset');
NEWyTest(idx) = -1;
idx = find(yTest == 'sunset');
NEWyTest(idx) = 1;
yTest = NEWyTest;

idx = find(testPredict == 'nonsunset');
NEWyTestPredict(idx) = -1;
idx = find(testPredict == 'sunset');
NEWyTestPredict(idx) = 1;
testPredict = NEWyTestPredict;

N = size(yTest,2);
falsePositives = sum((testPredict > 0)&(yTest < 0));
falseNegatives = sum((testPredict < 0)&(yTest > 0));
truePositives =  sum((testPredict > 0)&(yTest > 0));
trueNegatives =  sum((testPredict < 0)&(yTest < 0));
numCorrect = truePositives + trueNegatives;

TPR = truePositives/(truePositives+falseNegatives);
FPR = falsePositives/(trueNegatives+falsePositives);
precision = truePositives/(truePositives+falsePositives);
accuracy = numCorrect/N;

%% Find most wrong classifications
mostFalsePos = find(score(1:500, 2) > 0.99999);
mostFalseNeg = 500 + find(score(500:end, 1) > 0.999);
mostTruePos = 500 + find(score(500:end, 1) < 0.000000000000000001);
mostTrueNeg = find(score(1:500, 2) < 0.0000000001);