%% ME5411 Task 7 - Character Classification using MLP
% -----------------------------------------------------

clc; clear; close all;

dataDir = fullfile(pwd, 'dataset_2025'); % Change to actual path
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('? Dataset contains %d images, %d categories.\n', numel(imds.Files), numel(categories(imds.Labels)));

% Randomly shuffle
imds = shuffle(imds);

%% 2. Split into Training and Validation Sets
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.75, 'randomized');

%% 3. Define Preprocessing Function (Auto-detect Grayscale/RGB)
inputSize = [28 28]; % Adjustable to 32x32 or other sizes
imdsTrain.ReadFcn = @(x)preprocessImage(x, inputSize);
imdsVal.ReadFcn   = @(x)preprocessImage(x, inputSize);

%% 4. Convert to Feature Matrix
fprintf('? Building training data matrix...\n');
numTrain = numel(imdsTrain.Files);
numVal   = numel(imdsVal.Files);

XTrain = zeros(prod(inputSize), numTrain);
YTrain = imdsTrain.Labels;

for i = 1:numTrain
    img = readimage(imdsTrain, i);
    XTrain(:, i) = img(:);
end

XVal = zeros(prod(inputSize), numVal);
YVal = imdsVal.Labels;

for i = 1:numVal
    img = readimage(imdsVal, i);
    XVal(:, i) = img(:);
end

%% 5. Define MLP Network Structure
hiddenLayerSizes = [128 64]; % Adjustable hyperparameters
net = patternnet(hiddenLayerSizes);

% Training configuration
net.trainFcn = 'trainscg'; % Optimization algorithm (Scaled Conjugate Gradient)
net.trainParam.epochs = 50;
net.trainParam.lr = 0.001;
net.trainParam.showWindow = true;
net.performFcn = 'crossentropy';
net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.0;

%% 6. Train Network
fprintf('? Starting MLP training...\n');
targets = full(ind2vec(double(grp2idx(YTrain))'));
[net, tr] = train(net, XTrain, targets);

%% 7. Validation Set Prediction
fprintf('? Evaluating validation set...\n');
YValPred = net(XVal);
[~, predIdx] = max(YValPred, [], 1);
classNames = categories(YTrain);
YValPredLabel = categorical(classNames(predIdx));

accuracy = mean(YValPredLabel == YVal);
fprintf('? Validation Set Accuracy: %.2f%%\n', accuracy * 100);

%% 8. Plot Confusion Matrix (Compatible with older MATLAB versions)
fprintf('? Plotting confusion matrix...\n');
C = confusionmat(YVal, YValPredLabel);

%figure;
figure('Name','MLP Confusion Matrix', ...
       'NumberTitle','off', ...
       'Resize','on', ...              
       'Units','normalized', ...
       'OuterPosition',[0 0 1 1]);
imagesc(C);
colormap(jet);
colorbar;
title('MLP Validation Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');

% Display values on the matrix
textStrings = num2str(C(:), '%0.0f');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(C, 2), 1:size(C, 1));
hStrings = text(x(:), y(:), textStrings(:), ...
    'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9);
set(gca, 'XTick', 1:size(C, 2), 'YTick', 1:size(C, 1));
set(gca, 'XTickLabel', categories(YVal), 'YTickLabel', categories(YVal));
axis equal tight;


%% 9. Classification Report
tbl = table(YVal, YValPredLabel, 'VariableNames', {'TrueLabel', 'PredictedLabel'});
disp(tbl(1:10, :)); % Display first 10 results

%% 10. Save Model
save('mlp_model.mat', 'net');
fprintf('? Model saved to mlp_model.mat\n');

%% =================== Helper Functions ===================
function I = preprocessImage(filename, inputSize)
    % Read image
    I = imread(filename);

    % Convert to RGB if indexed image
    if ~isnumeric(I)
        I = ind2rgb(I, gray(256));
    end

    % Convert to grayscale if color image
    if ndims(I) == 3
        I = rgb2gray(I);
    end

    % Convert to double and normalize
    I = im2double(I);

    % Resize
    I = imresize(I, inputSize);
end