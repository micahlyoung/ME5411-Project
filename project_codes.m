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

% Question 3: Creating Sub-image
rect = [27.510000000000000,1.945100000000000e+02,9.449800000000000e+02,1.559800000000000e+02]
cropped_image = imcrop(A, rect);
figure;
imshow(uint8(cropped_image));
title("Cropped Image")

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

imageGray_cropped = rgb2gray(cropped_image);
segment_mask_v2 = intensity_slicing_v2(imageGray_cropped, 50, 100, 0, 1);
figure;
imshow(segment_mask_v2);
title("Binary Cropped Image");