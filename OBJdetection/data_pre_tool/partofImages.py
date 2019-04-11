import sys
import json
import cv2
import os
import shutil

datapath = '/private/luyujie/obstacle_detector/obstacle_detector/data/obstacle2d/JPGImages/'
trainpath = '/private/luyujie/obstacle_detector/obstacle_detector/data/obstacle2d/ImageSets/'
trainsetfile = 'val0629.txt'
outputpath = '/home/luyujie/SNIPER/data/coco'
with open(trainpath + trainsetfile) as f:
    cnt = 0
    for line in f:
        cnt += 1
        if cnt > 1000:
            break
        s = str(cnt).zfill(12)
        imagepath = os.path.join(datapath, line.strip() + '.jpg')
        newimgpath = os.path.join(outputpath, 'images/val20', 'COCO_val2014_'+s + '.jpg')
        shutil.copy(imagepath, newimgpath)
