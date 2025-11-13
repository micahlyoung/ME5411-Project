%% ===========================================================
%  ME5411 Task 7 - MLP Classifier
% ============================================================
clc; clear; close all;

%% ----------------------- Data Preparation -----------------------
dataDir = fullfile(pwd, 'dataset_2025');
if ~exist(dataDir, 'dir')
    error('Dataset folder not found. Please check the path.');
end

imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValandTest] = splitEachLabel(imds, 0.75, 'randomized');
[imdsVal, imdsTest] = splitEachLabel(imdsValandTest, 0.60, 'randomized'); 
inputSize = [32 32 1];

augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest = augmentedImageDatastore(inputSize, imdsTest);
numClasses = numel(categories(imds.Labels));

%% ----------------------- MLP Model Architecture -----------------------
layers = [
    imageInputLayer(inputSize, 'Normalization', 'zscore', 'Name','input')
    
    % Hidden Layer 1
    fullyConnectedLayer(512, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop1')

    % Hidden Layer 2
    fullyConnectedLayer(256, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'drop2')

    % Output Layer
    fullyConnectedLayer(numClasses, 'Name', 'fcOut')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
    ];

lgraph = layerGraph(layers);

%% ----------------------- Training Options -----------------------
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize',32, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress');

%% ----------------------- Train the Model -----------------------
net = trainNetwork(augTrain, lgraph, options);

%% ----------------------- Evaluate Performance -----------------------
YPred = classify(net, augTest);
YTrue = imdsTest.Labels;
acc = mean(YPred == YTrue);
fprintf('\nValidation Accuracy (MLP): %.2f%%\n', acc*100);

figure;
confusionchart(YTrue, YPred);
title('MLP Confusion Matrix');

save('mlp_model.mat','net');