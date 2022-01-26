%% Sunset Detector
% Jordan Asman and Cory Snyder
% Code built from some template code from Dr. Boutell

%% Extract training data
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
% yTrain = NEWyTrain;
% save('yTrain.mat')
% save('xTrain.mat')

%% Extract Testing Data
clc;

rootdir = 'SunsetImages/';
subdir = [rootdir 'test'];

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% Make datastores for the validation and testing sets similarly.

fprintf('Read images into datastores\n');

xTest = imageDatastoreReader(trainImages);
yTest = trainImages.Labels;

% idx = find(yTest == 'nonsunset');
% NEWyTest(idx) = -1;
% idx = find(yTest == 'sunset');
% NEWyTest(idx) = 1;
% yTest = NEWyTest;
% save('yTest.mat')
% save('xTest.mat')

%% Extract Validation Data
clc;

rootdir = 'SunsetImages/';
subdir = [rootdir 'validate'];

trainImages = imageDatastore(...
    subdir, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

% Make datastores for the validation and testing sets similarly.

fprintf('Read images into datastores\n');

xValidate = imageDatastoreReader(trainImages);
yValidate = trainImages.Labels;

% idx = find(yValidate == 'nonsunset');
% NEWyValidate(idx) = -1;
% idx = find(yValidate == 'sunset');
% NEWyValidate(idx) = 1;
% yValidate = NEWyValidate;
% save('yValidate.mat')
% save('xValidate.mat')

%% Train and evaluate an SVM
clear;clc;close all;

% Load in the training and validation Feature Vectors and Classifications
% from the .mat files
load('xTrain.mat');
load('yTrain.mat');
load('xValidate.mat');
load('yValidate.mat');
load('xTest.mat');
load('yTest.mat');
yTrain = yTrain';
yValidate = yValidate';
yTest = yTest';

% rng default
% net = fitcsvm(xTrain,yTrain,'Standardize',true,'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto', ...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
%     'expected-improvement-plus'))

kernelScale = 1154;%25.27
boxConstraint = 49;%4.26

net = fitcsvm(xTrain,yTrain,'KernelFunction','rbf', 'Standardize',false, ...
    'KernelScale',kernelScale,'ClassNames',[-1, 1], 'BoxConstraint',boxConstraint);

% Test the net on the Testing Set
[predictedClasses,distances] = predict(net,xValidate);

% Get the statistics of how the net performed
N = size(yValidate,1);
falsePositives = sum((predictedClasses > 0)&(yValidate < 0))
falseNegatives = sum((predictedClasses < 0)&(yValidate > 0))
truePositives =  sum((predictedClasses > 0)&(yValidate > 0))
trueNegatives =  sum((predictedClasses < 0)&(yValidate < 0))
numCorrect = truePositives + trueNegatives;

TPR = truePositives/(truePositives+falseNegatives)
FPR = falsePositives/(trueNegatives+falsePositives)
precision = truePositives/(truePositives+falsePositives)
accuracy = numCorrect/N

disp('Test Set ---------------------------------------------------------');

% Test the net on the Testing Set
[predictedClasses,distances] = predict(net,xTest);

% Get the statistics of how the net performed with different thresholds
thresholds = 0;%[-2:0.1:2];
for i = 1:max(size(thresholds))
  N = size(yTest,1);
  falsePositives = sum((distances(:,2) > thresholds(i))&(yTest < 0));
  falseNegatives = sum((distances(:,2) < thresholds(i))&(yTest > 0));
  truePositives =  sum((distances(:,2) > thresholds(i))&(yTest > 0));
  trueNegatives =  sum((distances(:,2) < thresholds(i))&(yTest < 0));
  numCorrect = truePositives + trueNegatives;

  TPR(i) = truePositives/(truePositives+falseNegatives);
  FPR(i) = falsePositives/(trueNegatives+falsePositives);
  precision = truePositives/(truePositives+falsePositives);
  accuracy(i) = numCorrect/N;
end

% Create a new figure. You can also number it: figure(1)
figure(1);
% Hold on means all subsequent plot data will be overlaid on a single plot
hold on;
% Plots using a blue line (see 'help plot' for shape and color codes 
plot(FPR, TPR, 'b-', 'LineWidth', 2);
% Overlaid with circles at the data points
plot(FPR, TPR, 'bo', 'MarkerSize', 6, 'LineWidth', 2);

% You could repeat here with a different color/style if you made 
% an enhancement and wanted to show that it outperformed the baseline.

% Title, labels, range for axes
title('MFE SVM ROC Curve', 'fontSize', 18);
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
% TPR and FPR range from 0 to 1. You can change these if you want to zoom in on part of the graph.
grid on;
axis([0 1 0 1]);
hold off;


% % Grid Search with the hyperparameters BoxConstraint and KernelScale
% 
% Cs = [1:1:20];%[10:10:200];%[10:10:200];
% sigmas = [1050:1:1100];%[900:10:1200];%[100:10:1000];
% Cs = [45:0.5:60];
% sigmas = [1150:1:1200];
% bestAccuracy = 0;
% bestC = 0;
% bestSigma = 0;
% for i = 1:size(Cs,2)
%     for j = 1:size(sigmas,2)
%         % Train the net for this set of hyperparameters
%         net = fitcsvm(xTrain,yTrain,'KernelFunction','rbf', 'Standardize',false, ...
%               'KernelScale',sigmas(j),'ClassNames',[-1, 1], 'BoxConstraint',Cs(i));
%         
%         % Use the net to classify the test set and get the accuracy
%         [predictedClasses,distances] = predict(net,xValidate);  
%           
%         N = size(yValidate,1);
%         falsePositives(i,j) = sum((predictedClasses > 0)&(yValidate < 0));
%         falseNegatives(i,j) = sum((predictedClasses < 0)&(yValidate > 0));
%         truePositives(i,j) =  sum((predictedClasses > 0)&(yValidate > 0));
%         trueNegatives(i,j) =  sum((predictedClasses < 0)&(yValidate < 0));
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
load('GridSearchResultsValidate6.mat','accuracy');
load('GridSearchResultsValidate6.mat','sigmas');
load('GridSearchResultsValidate6.mat','Cs');

figure();
mesh(Cs,sigmas,accuracy');
xlabel('Box Constraint: C');
ylabel('Kernel Scale: sigma');
zlabel('Accuracy');