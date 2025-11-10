% Question 1 of Project

filename = "charact2.bmp"; % Specifying the filename of the image
A = imread(filename); % Reading the image on the file
image(A); % Displaying the Image

[rows, cols, numChannels] = size(A);

disp(['Image dimensions: ', num2str(rows), ' rows, ', num2str(cols), ' columns, ', num2str(numChannels), ' color channels.']); % Printing the dimension of the image

% Performing standard grayscale histogram stretching

imageGray = rgb2gray(A);
colormap(gray(256))
title("Grayscale Histogram");
% imtool(imageGray);

% Performing Intensity slicing, we got ideas for implementation from the
% following matlab forum: https://ww2.mathworks.cn/matlabcentral/fileexchange/48847-intensity-level-slicing

function segment_mask = intensity_slicing(original_image, lower_threshold, upper_threshold)
    [rows, columns] = size(original_image);
    segment_mask = zeros(rows, columns); % Initialize the segmentation mask
    for i = 1:rows
        for j = 1:columns
            if (lower_threshold < original_image(i, j) && original_image(i, j) < upper_threshold)
                segment_mask(i, j) = 220;
            else
                segment_mask(i, j) = 35;
            end
        end
    end
end

segment_mask = intensity_slicing(imageGray, 50, 100);
figure;
imshow(imageGray);
figure;
imshow(uint8(segment_mask));

% Histogram Equalization using Matlab's histeq() function
J = histeq(imageGray);

%%
% Question 2 of Project

function filtered_image = convolution_filter(original_image, kernel_size)
    gray_image = rgb2gray(original_image);
    averaging_kernel = ones(kernel_size, kernel_size) / kernel_size^2;

    % Applying convolution
    filtered_image = conv2(gray_image, averaging_kernel, "same");
end

% Trying out the 5x5 averaging filter
filtered_image_5x5 = convolution_filter(A, 5);
figure;
imshow(uint8(filtered_image_5x5));
title("Averaging 5x5 Filtered Image");

% Trying out the 3x3 averaging filter
filtered_image_3x3 = convolution_filter(A, 3);
figure;
imshow(uint8(filtered_image_3x3));
title("Averaging 3x3 filtered Image");

% Trying otu the 8x8 averaging filter
filtered_image_8x8 = convolution_filter(A, 8);
figure;
imshow(uint8(filtered_image_8x8));
title("Averaging 8x8 filtered Image");

%%
% Question 3: Creating Sub-image
rect = [27.510000000000000,1.945100000000000e+02,9.449800000000000e+02,1.559800000000000e+02]
cropped_image = imcrop(A, rect);
figure;
imshow(uint8(cropped_image));
title("Cropped Image")

%%
% Question 4: Creating a binary image from the sub-image

% We modify the original segmentation function to allow for controllable
% output

function segment_mask_v2 = intensity_slicing_v2(original_image, lower_threshold, upper_threshold, low_output, high_output)
    [rows, columns] = size(original_image);
    segment_mask_v2 = zeros(rows, columns); % Initialize the segmentation mask
    for i = 1:rows
        for j = 1:columns
            if (lower_threshold < original_image(i, j) && original_image(i, j) < upper_threshold)
                segment_mask_v2(i, j) = high_output;
            else
                segment_mask_v2(i, j) = low_output;
            end
        end
    end
end

% imageGray_cropped = rgb2gray(cropped_image);
imageGray_cropped_filtered = convolution_filter(cropped_image, 11)
segment_mask_v2 = intensity_slicing_v2(imageGray_cropped_filtered, 50, 100, 0, 1);


figure;
imshow(segment_mask_v2);
title("Binary Cropped Image");

%%
% Question 5: Creating a set of boundary from the segmentation

% We would be using the bwboundaries function to do so
[boundaries, labels] = bwboundaries(segment_mask_v2, "holes");
imshow(label2rgb(labels, @jet, [.5, .5, .5]));
hold on
for k = 1:length(boundaries)
    boundary = boundaries{k};
    plot(boundary(:, 2), boundary(:, 1), "w", "lineWidth", 2)
end

%%
% Question 6: Combining all previous methods and experimenting values

final_filtered_image = convolution_filter(cropped_image, 9);
final_segmented_image = intensity_slicing_v2(final_filtered_image, 65, 115, 0, 1);
figure;
imshow(final_segmented_image);
title("Final Binary Cropped Image");

[final_boundaries, final_labels] = bwboundaries(final_segmented_image, "holes");
imshow(label2rgb(final_labels, @jet, [.5, .5, .5]));
hold on
for k = 1:length(final_boundaries)
    final_boundary = final_boundaries{k};
    plot(final_boundary(:, 2), final_boundary(:, 1), "w", "lineWidth", 2)
end


%% 
% Question 6: Splitting all the char images
BW = final_segmented_image;           
if mean(BW(:)) > 0.5
    BW = ~BW;                         
end

% denoise
BW = imclearborder(BW);
BW = bwareaopen(BW, 200);
% fill holes inside characters
BW_filled = imfill(BW, 'holes');

% connected components and region properties
CC    = bwconncomp(BW_filled);
stats = regionprops(CC, 'BoundingBox', 'Area', 'Solidity', 'Centroid');



widths = arrayfun(@(s) s.BoundingBox(3), stats);
heights = arrayfun(@(s) s.BoundingBox(4), stats);
avg_width = median(widths);
avg_height = median(heights);
bboxes = [];
for i = 1:length(stats)
    bb = stats(i).BoundingBox;
    w = bb(3);
    h = bb(4);
    % if a box is much wider than average, it may contain multiple characters
    if w > 1.5 * avg_width
        n_split = round(w / avg_width);
        split_width = w / n_split;
        % then, split this larger box horizontally
        for k = 0:n_split-1
            new_bb = [bb(1) + k * split_width, bb(2), split_width, h];
            bboxes = [bboxes; new_bb];
        end
    else
        bboxes = [bboxes; bb];
    end
end

[~, order] = sort(bboxes(:,1));
bboxes = bboxes(order,:);

figure; imshow(BW); title('Character mask');
hold on;
for i = 1:size(bboxes,1)
    rectangle('Position', bboxes(i,:), 'EdgeColor','r', 'LineWidth',1.5);
end
hold off;

output_folder = './segmentation_output/';
if ~exist(output_folder, 'dir'); mkdir(output_folder); end

for i = 1:size(bboxes,1)
    bb = bboxes(i,:);
    char_rgb = imcrop(BW, bb);
    char_rgb = imcomplement(char_rgb); 
    imwrite(char_rgb, fullfile(output_folder, sprintf('char_%02d.png', i)));
end

