import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def draw(filename):
    img = Image.open(filename)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    annopath = '/Users/c-ten/Desktop/aaaa.txt'
    with open(annopath) as atxt:
        annos = atxt.readlines()
    for ii, line in enumerate(annos):
        #print line
        parts = line.strip().split()
        
        x1 = float(parts[0])
        y1 = float(parts[1])
        x2 = float(parts[2]) #+ x1
        y2 = float(parts[3]) #+ y1
        olevel = float(parts[4])
        wid = max(0, x2 - x1)
        hei = max(0, y2 - y1)
        x1 = x1
        y1 = y1
        x2 = x2
        y2 = y2
        draw.line([(x1, y1), (x2, y1)], fill=(0, 0, 255))
        draw.line([(x2, y1), (x2, y2)], fill=(0, 0, 255))
        draw.line([(x2, y2), (x1, y2)], fill=(0, 0, 255))
        draw.line([(x1, y2), (x1, y1)], fill=(0, 0, 255))
w    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    draw('/Users/c-ten/Desktop/aaaa.jpg')
