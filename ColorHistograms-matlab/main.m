clear, clc, close all;

% this script runs benchmarks to find mean processing time per pixel

large = '../data/spotted_ball_3500.png';
disp(['Loading ' large]);
large_image = imread(large);
large_info = imfinfo(large);
large_bit_depth = large_info.BitDepth;
disp(['Processing ' large]);
[large_total, large_per_image, large_per_pixel] = hist_benchmark(large_image, large_bit_depth, 16, 4);

small = '../data/bstar100.png';
disp(['Loading ' small]);
small_image = imread(small);
small_info = imfinfo(small);
small_bit_depth = small_info.BitDepth;
disp(['Processing ' small]);
[small_total, small_per_image, small_per_pixel] = hist_benchmark(small_image, small_bit_depth, 16, 4900);

% display results in readable format
fprintf('File                            Total    Per image Per pixel\n');
fprintf('%-32s%-9.5f%-10.6f%e\n', large, large_total, large_per_image, large_per_pixel);
fprintf('%-32s%-9.5f%-10.6f%e\n', small, small_total, small_per_image, small_per_pixel);
