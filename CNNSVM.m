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

idx = find(yCNNTrain == 'nonsunset');
NEWyCNNTrain(idx) = -1;
idx = find(yCNNTrain == 'sunset');
NEWyCNNTrain(idx) = 1;
yCNNTrain = NEWyCNNTrain;
save('CNNTrain.mat','yCNNTrain','xCNNTrain')

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
yCNNValidate = validateImages.Labels;

idx = find(yCNNValidate == 'nonsunset');
NEWyCNNValidate(idx) = -1;
idx = find(yCNNValidate == 'sunset');
NEWyCNNValidate(idx) = 1;
yCNNValidate = NEWyCNNValidate;
save('CNNValidate.mat','yCNNValidate','xCNNValidate')

%% Extract Test Data
clc; clear all;

rootdir = 'SunsetImages/';
subdir = [rootdir 'test'];

fprintf('Read images into datastores\n');

testImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

net = alexnet;
inputSize = net.Layers(1).InputSize;
resizedTestImgs = augmentedImageDatastore(inputSize(1:2),testImages);
layer = 'fc7';

xCNNTest = activations(net,resizedTestImgs,layer,'OutputAs','rows');
yCNNTest = testImages.Labels;

idx = find(yCNNTest == 'nonsunset');
NEWyCNNTest(idx) = -1;
idx = find(yCNNTest == 'sunset');
NEWyCNNTest(idx) = 1;
yCNNTest = NEWyCNNTest;
save('CNNTest.mat','yCNNTest','xCNNTest')

%% Train and evaluate an SVM
clear all;clc;close all;

% Load in the training and validation Feature Vectors and Classifications
% from the .mat files
load('CNNTrain.mat');
load('CNNValidate.mat');
load('CNNTest.mat');
yCNNTrain = yCNNTrain';
yCNNValidate = yCNNValidate';
yCNNTest = yCNNTest';

% rng default
% net1 = fitcsvm(xCNNTrain,yCNNTrain,'Standardize',true,'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto', ...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
%     'expected-improvement-plus'))

kernelScale = 350;%75.13
boxConstraint = 10;%185.33
net = fitcsvm(xCNNTrain,yCNNTrain,'KernelFunction','rbf', 'Standardize',false, ...
    'KernelScale',kernelScale,'ClassNames',[-1, 1], 'BoxConstraint',boxConstraint);

% Test the net on the Testing Set
[predictedClasses,distances] = predict(net,xCNNValidate);

% Get the statistics of how the net performed
N = size(yCNNValidate,1);
falsePositives = sum((predictedClasses > 0)&(yCNNValidate < 0))
falseNegatives = sum((predictedClasses < 0)&(yCNNValidate > 0))
truePositives =  sum((predictedClasses > 0)&(yCNNValidate > 0))
trueNegatives =  sum((predictedClasses < 0)&(yCNNValidate < 0))
numCorrect = truePositives + trueNegatives;

TPR = truePositives/(truePositives+falseNegatives)
FPR = falsePositives/(trueNegatives+falsePositives)
precision = truePositives/(truePositives+falsePositives)
accuracy = numCorrect/N

disp('Test Set ---------------------------------------------------------');

% Test the net on the Testing Set
[predictedClasses,distances] = predict(net,xCNNTest);

% % Get the statistics of how the net performed with different thresholds
% thresholds = [-2:0.1:2];
% for i = 1:max(size(thresholds))
%   N = size(yCNNTest,1);
%   falsePositives = sum((distances(:,2) > thresholds(i))&(yCNNTest < 0));
%   falseNegatives = sum((distances(:,2) < thresholds(i))&(yCNNTest > 0));
%   truePositives =  sum((distances(:,2) > thresholds(i))&(yCNNTest > 0));
%   trueNegatives =  sum((distances(:,2) < thresholds(i))&(yCNNTest < 0));
%   numCorrect = truePositives + trueNegatives;
% 
%   TPR(i) = truePositives/(truePositives+falseNegatives);
%   FPR(i) = falsePositives/(trueNegatives+falsePositives);
%   precision = truePositives/(truePositives+falsePositives);
%   accuracy(i) = numCorrect/N;
% end
% 
% % Create a new figure. You can also number it: figure(1)
% figure(1);
% % Hold on means all subsequent plot data will be overlaid on a single plot
% hold on;
% % Plots using a blue line (see 'help plot' for shape and color codes 
% plot(FPR, TPR, 'b-', 'LineWidth', 2);
% % Overlaid with circles at the data points
% plot(FPR, TPR, 'bo', 'MarkerSize', 6, 'LineWidth', 2);
% 
% % You could repeat here with a different color/style if you made 
% % an enhancement and wanted to show that it outperformed the baseline.
% 
% % Title, labels, range for axes
% title('CFE SVM ROC Curve from Threshold Variance', 'fontSize', 18);
% xlabel('False Positive Rate', 'fontWeight', 'bold');
% ylabel('True Positive Rate', 'fontWeight', 'bold');
% % TPR and FPR range from 0 to 1. You can change these if you want to zoom in on part of the graph.
% grid on;
% axis([0 1 0 1]);
% hold off;

% % Grid Search with the hyperparameters BoxConstraint and KernelScale
% 
% Cs = [8:1:12];
% sigmas = [340:1:360];
% bestAccuracy = 0;
% bestC = 0;
% bestSigma = 0;
% for i = 1:size(Cs,2)
%     for j = 1:size(sigmas,2)
%         % Train the net for this set of hyperparameters
%         net = fitcsvm(xCNNTrain,yCNNTrain,'KernelFunction','rbf', 'Standardize',false, ...
%               'KernelScale',sigmas(j),'ClassNames',[-1, 1], 'BoxConstraint',Cs(i));
%         
%         % Use the net to classify the test set and get the accuracy
%         [predictedClasses,distances] = predict(net,xCNNValidate);  
%           
%         N = size(yCNNValidate,1);
%         falsePositives(i,j) = sum((predictedClasses > 0)&(yCNNValidate < 0));
%         falseNegatives(i,j) = sum((predictedClasses < 0)&(yCNNValidate > 0));
%         truePositives(i,j) =  sum((predictedClasses > 0)&(yCNNValidate > 0));
%         trueNegatives(i,j) =  sum((predictedClasses < 0)&(yCNNValidate < 0));
%         numCorrect(i,j) = truePositives(i,j) + trueNegatives(i,j);
%         accuracy(i,j) = numCorrect(i,j)/N;
%         fprintf('C Iter: %2i   sigma Iter: %2i\n', [i, j]');
%         if(accuracy(i,j) > bestAccuracy)
%             bestAccuracy = accuracy(i,j);
%             bestC = Cs(i);
%             bestSigma = sigmas(j);
%         end
%     end
% end
% 
% disp('Best Accuracy was found using the hyperparameters:');
% fprintf('C: %2d   sigma: %2d\n', [bestC, bestSigma]');

%% Show Grid Search Accuracies
close all;clc;clear;
load('CNNSVMGridSearchResults4.mat','accuracy');
load('CNNSVMGridSearchResults4.mat','sigmas');
load('CNNSVMGridSearchResults4.mat','Cs');

figure();
mesh(Cs,sigmas,accuracy');
xlabel('Box Constraint: C');
ylabel('Kernel Scale: sigma');
zlabel('Accuracy');

%% Plot the ROC curve for the MFE and CFE
% Create a new figure. You can also number it: figure(1)
figure();
% Hold on means all subsequent plot data will be overlaid on a single plot
hold on;
% Plots using a blue line (see 'help plot' for shape and color codes 
plot(MFEFPR, MFETPR, 'b-', 'LineWidth', 2);
plot(CFEFPR, CFETPR, 'r-', 'LineWidth', 2);
% Overlaid with circles at the data points
plot(MFEFPR, MFETPR, 'bo', 'MarkerSize', 6, 'LineWidth', 2);
plot(CFEFPR, CFETPR, 'ro', 'MarkerSize', 6, 'LineWidth', 2);

% You could repeat here with a different color/style if you made 
% an enhancement and wanted to show that it outperformed the baseline.

% Title, labels, range for axes
title('MFE vs. CFE SVM ROC Curve', 'fontSize', 18);
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
legend('MFE SVM','CFE SVM');

% TPR and FPR range from 0 to 1. You can change these if you want to zoom in on part of the graph.
grid on;
axis([0 1 0 1]);
hold off;