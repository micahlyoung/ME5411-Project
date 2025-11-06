%% ===========================================================
%  ME5411 Task 7 - ViT-like CNN Classifier
% ============================================================
clc; clear; close all;

%% ----------------------- Data Preparation -----------------------
dataDir = fullfile(pwd, 'dataset_2025');
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.75, 'randomized');
inputSize = [64 64 3];

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'gray2rgb');
numClasses = numel(categories(imds.Labels));

%% ----------------------- CNN-based ViT-like Model -----------------------
layers = [
    imageInputLayer(inputSize, 'Normalization', 'zscore', 'Name','input')

    % Patch extraction (simulate ViT patch embedding)
    convolution2dLayer(8, 64, 'Stride', 8, 'Padding', [0 0 0 0], 'Name','patchConv')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')

    % Transformer-like local mixing (using standard convolutions)
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name','mix1')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name','mix2')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')

    % Global pooling and classifier
    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(256, 'Name','fc1')
    reluLayer('Name','relu_fc')
    dropoutLayer(0.3, 'Name','dropout')
    fullyConnectedLayer(numClasses, 'Name','fcOut')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
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
YPred = classify(net, augVal);
YTrue = imdsVal.Labels;
acc = mean(YPred == YTrue);
fprintf('\nValidation Accuracy (CNN-based ViT): %.2f%%\n', acc*100);

figure;
confusionchart(YTrue, YPred);
title('CNN-based ViT Confusion Matrix');

save('vit_like_cnn_model.mat','net');
