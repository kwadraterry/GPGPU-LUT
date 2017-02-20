function hist = histogram( img, num_bins )
%HISTOGRAM Computes c-dimensional color histogram of an image
%   c is number of channels in an image
%   arguments:
%      img - input image matrix (w*h*c uint8)
%      num_bins - number of bins in the histogram
%   return hist - num_bins*c color histogram

[w,h,c] = size(img);
hist = zeros(num_bins, c);

% total number of possible values for each channel is assumed to be 256
bin_size = floor(256 / num_bins);

for x=1:w
    for y=1:h
        for z=1:c
            % compute the bin that the value in (x, y, z) belongs to
            bin = floor(double(img(x,y,z)) / bin_size) + 1;
            % increment the bin
            hist(bin, z) = hist(bin, z) + 1;
        end;
    end;
end;
end

