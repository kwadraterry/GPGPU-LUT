import pycuda.autoinit
import pycuda.driver as drv
import numpy
from scipy import misc
from color_histogram_cuda_module import histogram_atomics, histogram_accum

def histogram(image_path, num_bins):
    image = misc.imread(image_path)

    bin_size = 256 / num_bins

    # calculate image dimensions
    (w, h, c) = image.shape

    # reinterpret image with 4-byte type
    image = image.view(numpy.uint32)

    dest = numpy.zeros((bin_size, c), numpy.uint32)
    parts = num_bins * c
    block1 = (32, 4, 1)
    grid1 = (16, 16, 1)
    partial = numpy.zeros(grid1[0] * grid1[1] * parts, numpy.uint32)
    block2 = (128,1, 1)
    grid2 = ((c * num_bins + block2[0] - 1) / block2[0], 1, 1)

    histogram_atomics(drv.In(image), w, h, drv.Out(partial), block=block1, grid=grid1)
    histogram_accum(drv.In(partial), grid1[0] * grid1[1], drv.Out(dest))

    return dest
