import sys
import json
import cv2
import os
import shutil

mapannopath = '/home/luyujie/map_anno'
trainpath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/ImageSets/'
datapath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages'
datapath2 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages2'
datapath3 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages_crossbar'
annopath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/Annotations/'
trainsetfile = 'train1030.txt'
outputpath = '/home/luyujie/SNIPER/data/coco'
phase = 'train'

with open(trainpath + trainsetfile) as f:
    vis_ext = '.txt'
    count = 0
    for line in f:
        imagepath = os.path.join(datapath, line.strip() + '.jpg')
        # no obstacle currently drop it
        txtpath = os.path.join(annopath, line.strip() + '.txt')
        maptxtpath = os.path.join(mapannopath, line.strip() + '.txt')
        oritxtpath = os.path.join('/home/luyujie/imid_anno','{}{}'.format(count, vis_ext))

        if not os.path.exists(txtpath):
            print txtpath
            continue
        #print imagepath
        if not os.path.exists(imagepath):
            imagepath = os.path.join(datapath2, line.strip() + '.jpg')
            if not os.path.exists(imagepath):
                imagepath = os.path.join(datapath3, line.strip() + '.jpg')
        if not os.path.exists(imagepath):
            continue

        #create dir
        last_position = -1
        while True:
            position = line.find("/", last_position + 1)
            if position == -1:
                break
            last_position = position
        dirpath = os.path.join(mapannopath, line.strip()[0:last_position])
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

        #write map txt
        '''
        fobj = open(maptxtpath, 'a')
        with open(oritxtpath) as oritxt:
            for line in oritxt:
                fobj.write(line)
        fobj.close()
        '''

        shutil.copy(oritxtpath, maptxtpath)
        count += 1
