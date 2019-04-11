import sys
import json
import cv2
import os
import shutil
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

checkVisDir = '/home/luyujie/check_val0629vis'
checkOriDir = '/home/luyujie/check_val0629ori'
#checkImgDir = '/home/luyujie/SNIPER/data/coco/images/test2015'
trainpath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/ImageSets/'
datapath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages'
datapath2 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages2'
datapath3 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages_crossbar'
annopath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/Annotations/'
trainsetfile = 'val0629.txt'

def draw(filename, txtpath, imgname):
    dpi = 80
    issave = False
    img = Image.open(filename)
    #w, h = img.size
    #draw = ImageDraw.Draw(img)
    with open(txtpath) as atxt:
        annos = atxt.readlines()
    # Create a canvas the same size of the image
    height = 1200
    width = 1920
    out_size = width/float(dpi), height/float(dpi)
    fig = plt.figure(figsize=out_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    issaveori = True
    # Display the image
    ax.imshow(img, interpolation='nearest')
    for ii, line in enumerate(annos):
        #print line
        color = (random.random(), random.random(), random.random())
        parts = line.strip().split()
        if not parts[5].__contains__('person'):continue
        x1 = float(parts[0])
        y1 = float(parts[1])
        x2 = float(parts[2]) #+ x1
        y2 = float(parts[3]) #+ y1
        olevel = float(parts[4])
        wid = max(0, x2 - x1)
        hei = max(0, y2 - y1)
        if issaveori:
            issaveori = False
            fig.savefig(os.path.join(checkOriDir, imgname + '_ori.jpg'), dpi=dpi, transparent=True)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=3.5)
        ax.add_patch(rect)
        ax.text(x1, y1 - 2 if y1-2 > 15 else y1+15, '{:s} {:.1f} {:.1f}'.format(parts[5], 1, olevel), bbox=dict(facecolor=color, alpha=0.5), fontsize=10, color='white')
        issave = True
        #draw.line([(x1, y1), (x2, y1)], fill=(0, 0, 255))
        #draw.line([(x2, y1), (x2, y2)], fill=(0, 0, 255))
        #draw.line([(x2, y2), (x1, y2)], fill=(0, 0, 255))
        #draw.line([(x1, y2), (x1, y1)], fill=(0, 0, 255))
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    if issave:
        fig.savefig(os.path.join(checkVisDir, imgname + '_anno.jpg'), dpi=dpi, transparent=True)
    '''
    x1 = 44.0
    y1 = 264.0
    x2 = 201.0 + x1
    y2 = 80.0 + y1
    draw.line([(x1, y1), (x2, y1)], fill=(0, 0, 255))
    draw.line([(x2, y1), (x2, y2)], fill=(0, 0, 255))
    draw.line([(x2, y2), (x1, y2)], fill=(0, 0, 255))
    draw.line([(x1, y2), (x1, y1)], fill=(0, 0, 255))
    '''
    plt.cla()
    plt.clf()
    plt.close()
    #plt.imshow(img)
    #plt.show()

with open(trainpath + trainsetfile) as f:
    cnt = 0
    for line in f:
        '''
        if line[0:8] > '20181000':
            cnt += 1
        else:
            continue
        '''
        cnt += 1
        if cnt > 1000:
            break
        imagepath = os.path.join(datapath, line.strip() + '.jpg')
        if not os.path.exists(imagepath):
            imagepath = os.path.join(datapath2, line.strip() + '.jpg')
            if not os.path.exists(imagepath):
                imagepath = os.path.join(datapath3, line.strip() + '.jpg')
        if not os.path.exists(imagepath):
            continue

        txtpath = os.path.join(annopath, line.strip() + '.txt')
        if not os.path.exists(txtpath):
            print txtpath
            continue
        last_position = -1
        while True:
            position = line.find("/", last_position + 1)
            if position == -1:
                break
            last_position = position
        imgname = line[last_position+1:]
        #newimgpath = os.path.join('/home/luyujie/check_val1010', imgname + '.jpg')
        #shutil.copy(imagepath, newimgpath)
        draw(imagepath, txtpath, imgname)