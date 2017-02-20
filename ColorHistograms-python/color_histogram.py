from PIL import Image
from collections import defaultdict
import sys

im = Image.open(sys.argv[1])
colors = defaultdict(int)
for pixel in im.getdata():
    colors[pixel] += 1
print colors
