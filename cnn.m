clc; 
clear; 
%% load data 
path = './dataset_2025'; 

imds = imageDatastore(path, "IncludeSubfolders", true, ...
    'LabelSource', 'foldernames'); 

[imdsTrain, imdsValandTest] = splitEachLabel(imds, 0.75, 'randomized'); 
[imdsVal, imdsTest] = splitEachLabel(imdsValandTest, 0.60, 'randomized'); 

%% visualization
% figure; 
% perm = randperm(numel(imds.Files), 20); 
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
%     title(char(imds.Labels(perm(i)))); 
%     drawnow;
% end

%% preprocessing
inputSize = [32 32 1]; 

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsVal = augmentedImageDatastore(inputSize, imdsVal);
augimdsTest = augmentedImageDatastore(inputSize, imdsTest);

%% define network
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(3, 4, 'Padding','same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    convolution2dLayer(3, 16, 'Padding','same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    convolution2dLayer(3, 32, 'Padding','same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

% analyzeNetwork(layers);
%% train
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',50, ...
    'MiniBatchSize', 128, ...           
    'Shuffle','every-epoch', ...       
    'ValidationData',augimdsVal, ...
    'ValidationFrequency', 50, ...
    'Verbose',true, ...
    'Plots','none', ...
    'OutputFcn', @train_val_visualization.plotTrainingProgress); 
[net, trainInfo] = trainNetwork(augimdsTrain, layers, options);

%% 
save('CNN_output/trained_net_cnn.mat', 'net');
save('CNN_output/training_info_cnn.mat', 'trainInfo');

%% scoring
[YPred, scores] = classify(net, augimdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);

figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for Test Set');


%% test for the given image
test_path = './outputs/segmentation_output/';  
fileList = dir(fullfile(test_path, 'char_*.png'));  

figure('Name','Predicted Characters','NumberTitle','off');
numImages = numel(fileList);

load('outputs/CNN_output/trained_net_cnn.mat', 'net');

cols = numImages;                           
rows = 1; 

for i = 1:numImages
    img = imread(fullfile(test_path, fileList(i).name));
    if size(img,3)==3
        img = rgb2gray(img);
    end
    img = padSquare(img, 32) * 255;     
    img = reshape(img,[32 32 1]);    
    
    [label, score] = classify(net, img);

    subplot(rows, cols, i);
    imshow(img, []);
    title(sprintf('%s\n(%.2f%%)', string(label), max(score)*100), ...
          'FontSize', 10);
end

sgtitle('Predicted Characters from Segmentation Output');


function out = padSquare(img, targetSize, marginRatio)
    if nargin < 3
        marginRatio = 0.1; 
    end

    if size(img,3) == 3
        img = rgb2gray(img);
    end

    if isfloat(img)
        whiteVal = 1;
    elseif islogical(img)
        whiteVal = true;
    else
        whiteVal = intmax(class(img));
    end

    [h, w] = size(img);

    margin = round(max(h, w) * marginRatio);

    newSide = max(h, w) + 2 * margin;

    canvas = repmat(whiteVal, [newSide, newSide]);

    yStart = floor((newSide - h)/2) + 1;
    xStart = floor((newSide - w)/2) + 1;

    canvas(yStart:yStart+h-1, xStart:xStart+w-1) = img;

    out = imresize(canvas, [targetSize targetSize]);

    out = reshape(out, [targetSize targetSize 1]);
end




