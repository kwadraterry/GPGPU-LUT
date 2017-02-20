function [total_time, time_per_image, time_per_pixel] = hist_benchmark( img, bins, repeat )
%HIST_BENCHMARK Measures time to compute histogram of an image
%   Test is repeated <repeat> times
%   Arguments:
%       img - image to process
%       bins - number of bins in the histogram
%       repeat - number of times to repeat the test
%   Returns:
%       total_time - total time to run the tests
%       time_per_image - mean single test time
%       time_per_pixel - mean time to process one pixel of an image

[W, H, ~] = size(img); % image parameters;
pixels_computed = W * H * repeat;

tic;
for i=1:repeat
    histogram(img, bins);
end;
total_time = toc;
time_per_image = total_time / repeat;
time_per_pixel = total_time / pixels_computed;

end

