clear, clc, close all;

% run benchmarks to find mean processing time per pixel

large_image = imread('../data/spotted_ball_3500.png');
[large_total, large_per_image, large_per_pixel] = hist_benchmark(large_image, 16, 4);

small_image = imread('../data/bstar100.png');
[small_total, small_per_image, small_per_pixel] = hist_benchmark(small_image, 16, 4900);
