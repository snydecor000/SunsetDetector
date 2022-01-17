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

idx = find(yTest == 'nonsunset');
NEWyTest(idx) = -1;
idx = find(yTest == 'sunset');
NEWyTest(idx) = 1;
yTest = NEWyTest;
save('yTest.mat')
save('xTest.mat')

%% Train and evaluate an SVM
clear;clc;close all;
load('xTrain.mat');
load('yTrain.mat');
load('xTest.mat');
load('yTest.mat');
% rng default
% net = fitcsvm(xTrain,yTrain,'Standardize',true,'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto', ...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
%     'expected-improvement-plus'))

kernelScale = 600;%25.27
boxConstraint = 100;%4.26
net = fitcsvm(xTrain,yTrain,'KernelFunction','rbf', 'Standardize',false, ...
    'KernelScale',kernelScale,'ClassNames',[-1, 1], 'BoxConstraint',boxConstraint);

% Test the net on the Testing Set
[predictedClasses,distances] = predict(net,xTest);

% Get the statistics of how the net performed
N = size(yTest,2);
falsePositives = sum((predictedClasses > 0)&(yTest' < 0))
falseNegatives = sum((predictedClasses < 0)&(yTest' > 0))
truePositives =  sum((predictedClasses > 0)&(yTest' > 0))
trueNegatives =  sum((predictedClasses < 0)&(yTest' < 0))
numCorrect = truePositives + trueNegatives;

TPR = truePositives/(truePositives+falseNegatives)
FPR = falsePositives/(trueNegatives+falsePositives)
precision = truePositives/(truePositives+falsePositives)
accuracy = numCorrect/N


% Grid Search with the hyperparameters BoxConstraint and KernelScale

Cs = [1:1:20];%[10:10:200];%[10:10:200];
sigmas = [1050:1:1100];%[900:10:1200];%[100:10:1000];
bestAccuracy = 0;
bestC = 0;
bestSigma = 0;
for i = 1:size(Cs,2)
    for j = 1:size(sigmas,2)
        % Train the net for this set of hyperparameters
        net = fitcsvm(xTrain,yTrain,'KernelFunction','rbf', 'Standardize',false, ...
              'KernelScale',sigmas(j),'ClassNames',[-1, 1], 'BoxConstraint',Cs(i));
        
        % Use the net to classify the test set and get the accuracy
        [predictedClasses,distances] = predict(net,xTest);  
          
        N = size(yTest',1);
        falsePositives(i,j) = sum((predictedClasses > 0)&(yTest' < 0));
        falseNegatives(i,j) = sum((predictedClasses < 0)&(yTest' > 0));
        truePositives(i,j) =  sum((predictedClasses > 0)&(yTest' > 0));
        trueNegatives(i,j) =  sum((predictedClasses < 0)&(yTest' < 0));
        numCorrect(i,j) = truePositives(i,j) + trueNegatives(i,j);
        accuracy(i,j) = numCorrect(i,j)/N;
        fprintf('C Iter: %2i   sigma Iter: %2i\n', [i, j]');
        if(accuracy(i,j) > bestAccuracy)
            bestAccuracy = accuracy(i,j);
            bestC = Cs(i);
            bestSigma = sigmas(j);
        end
    end
end

disp('Best Accuracy was found using the hyperparameters:');
fprintf('C: %2d   sigma: %2d\n', [bestC, bestSigma]');

%% Show Grid Search Accuracies
close all;clc;clear;
load('GridSearchResults4.mat','accuracy');
Cs = [1:1:20];%[10:10:200];%[10:10:200];
sigmas = [1050:1:1100];%[900:10:1200];%[100:10:1000];

figure();
mesh(Cs,sigmas,accuracy');
xlabel('Box Constraints');
ylabel('Kernel Scale: sigma');
zlabel('Accuracy');