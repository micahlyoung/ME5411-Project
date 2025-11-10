%% test for the given image
test_path = './segmentation_output/';  
fileList = dir(fullfile(test_path, 'char_*.png'));  

figure('Name','Predicted Characters','NumberTitle','off');
numImages = numel(fileList);

load('CNN_output/trained_net_cnn.mat', 'net');

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




%%
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