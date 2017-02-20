from PIL import Image
from collections import defaultdict
import sys

def cpu_hist(image):
    im = Image.open(sys.argv[1])
    colors = defaultdict(int)
    for pixel in im.getdata():
        colors[pixel] += 1
    print colors
