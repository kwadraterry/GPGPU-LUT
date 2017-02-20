import sys
from color_histogram_cuda import histogram

print histogram(sys.argv[1], 16)
