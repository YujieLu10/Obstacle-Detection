import sys
import json
import cv2
import os
import shutil

dataset = { 'info': {
            'description': "This is stable 1.0 version of the 2014 MS COCO dataset.", 
            'url': "http://mscoco.org", 
            'version': "1.0", 
            'year': 2018, 
            'contributor': "FABU Group", 
            'date_created': "2018-10-19 22:00:00.357475"}, 
            'images':[],
            'annotations':[],
            'categories': [
            {'supercategory:': 'car', 'id': 1, 'name': 'car'},
            {'supercategory:': 'bus', 'id': 2, 'name': 'bus'},
            {'supercategory:': 'truck', 'id': 3, 'name': 'truck'},
            {'supercategory:': 'person', 'id': 4, 'name': 'person'},
            {'supercategory:': 'bicycle', 'id': 5, 'name': 'bicycle'},
            {'supercategory:': 'tricycle', 'id': 6, 'name': 'tricycle'},
            {'supercategory:': 'block', 'id': 7, 'name': 'block'}]
        }
'''
dataset = { 'info': {
            'description': "This is stable 1.0 version of the 2014 MS COCO dataset.", 
            'url': "http://mscoco.org", 
            'version': "1.0", 
            'year': 2018, 
            'contributor': "FABU Group", 
            'date_created': "2018-10-19 22:00:00.357475"}, 
            'images':[],
            'annotations':[],
            'categories': [
            {'supercategory:': 'person', 'id': 1, 'name': 'person'},
            {'supercategory:': 'bicycle', 'id': 2, 'name': 'bicycle'},
            {'supercategory:': 'car', 'id': 3, 'name': 'car'},
            {'supercategory:': 'motorcycle', 'id': 4, 'name': 'motorcycle'},
            {'supercategory:': 'airplane', 'id': 5, 'name': 'airplane'},
            {'supercategory:': 'bus', 'id': 6, 'name': 'bus'},
            {'supercategory:': 'train', 'id': 7, 'name': 'train'},
            {'supercategory:': 'truck', 'id': 8, 'name': 'truck'},
            {'supercategory:': 'boat', 'id': 9, 'name': 'boat'},
            {'supercategory:': 'traffic light', 'id': 10, 'name': 'traffic light'},
            {'supercategory:': 'fire hydrant', 'id': 11, 'name': 'fire hydrant'},
            {'supercategory:': 'stop sign', 'id': 12, 'name': 'stop sign'},
            {'supercategory:': 'parking meter', 'id': 13, 'name': 'parking meter'},
            {'supercategory:': 'bench', 'id': 14, 'name': 'bench'},
            {'supercategory:': 'bird', 'id': 15, 'name': 'bird'},
            {'supercategory:': 'cat', 'id': 16, 'name': 'cat'},
            {'supercategory:': 'dog', 'id': 17, 'name': 'dog'},
            {'supercategory:': 'horse', 'id': 18, 'name': 'horse'},
            {'supercategory:': 'sheep', 'id': 19, 'name': 'sheep'},
            {'supercategory:': 'cow', 'id': 20, 'name': 'cow'},
            {'supercategory:': 'elephant', 'id': 21, 'name': 'elephant'},
            {'supercategory:': 'bear', 'id': 22, 'name': 'bear'},
            {'supercategory:': 'zebra', 'id': 23, 'name': 'zebra'},
            {'supercategory:': 'giraffe', 'id': 24, 'name': 'giraffe'},
            {'supercategory:': 'backpack', 'id': 25, 'name': 'backpack'},
            {'supercategory:': 'umbrella', 'id': 26, 'name': 'umbrella'},
            {'supercategory:': 'handbag', 'id': 27, 'name': 'handbag'},
            {'supercategory:': 'tie', 'id': 28, 'name': 'tie'},
            {'supercategory:': 'suitcase', 'id': 29, 'name': 'suitcase'},
            {'supercategory:': 'frisbee', 'id': 30, 'name': 'frisbee'},
            {'supercategory:': 'skis', 'id': 31, 'name': 'skis'},
            {'supercategory:': 'snowboard', 'id': 32, 'name': 'snowboard'},
            {'supercategory:': 'sports\nball', 'id': 33, 'name': 'sports\nball'},
            {'supercategory:': 'kite', 'id': 34, 'name': 'kite'},
            {'supercategory:': 'baseball\nbat', 'id': 35, 'name': 'baseball\nbat'},
            {'supercategory:': 'baseball glove', 'id': 36, 'name': 'baseball glove'},
            {'supercategory:': 'skateboard', 'id': 37, 'name': 'skateboard'},
            {'supercategory:': 'surfboard', 'id': 38, 'name': 'surfboard'},
            {'supercategory:': 'tennis racket', 'id': 39, 'name': 'tennis racket'},
            {'supercategory:': 'bottle', 'id': 40, 'name': 'bottle'},
            {'supercategory:': 'wine\nglass', 'id': 41, 'name': 'wine\nglass'},
            {'supercategory:': 'cup', 'id': 42, 'name': 'cup'},
            {'supercategory:': 'fork', 'id': 43, 'name': 'fork'},
            {'supercategory:': 'knife', 'id': 44, 'name': 'knife'},
            {'supercategory:': 'spoon', 'id': 45, 'name': 'spoon'},
            {'supercategory:': 'bowl', 'id': 46, 'name': 'bowl'},
            {'supercategory:': 'banana', 'id': 47, 'name': 'banana'},
            {'supercategory:': 'apple', 'id': 48, 'name': 'apple'},
            {'supercategory:': 'sandwich', 'id': 49, 'name': 'sandwich'},
            {'supercategory:': 'orange', 'id': 50, 'name': 'orange'},
            {'supercategory:': 'broccoli', 'id': 51, 'name': 'broccoli'},
            {'supercategory:': 'carrot', 'id': 52, 'name': 'carrot'},
            {'supercategory:': 'hot dog', 'id': 53, 'name': 'hot dog'},
            {'supercategory:': 'pizza', 'id': 54, 'name': 'pizza'},
            {'supercategory:': 'donut', 'id': 55, 'name': 'donut'},
            {'supercategory:': 'cake', 'id': 56, 'name': 'cake'},
            {'supercategory:': 'chair', 'id': 57, 'name': 'chair'},
            {'supercategory:': 'couch', 'id': 58, 'name': 'couch'},
            {'supercategory:': 'potted plant', 'id': 59, 'name': 'potted plant'},
            {'supercategory:': 'bed', 'id': 60, 'name': 'bed'},
            {'supercategory:': 'dining table', 'id': 61, 'name': 'dining table'},
            {'supercategory:': 'toilet', 'id': 62, 'name': 'toilet'},
            {'supercategory:': 'tv', 'id': 63, 'name': 'tv'},
            {'supercategory:': 'laptop', 'id': 64, 'name': 'laptop'},
            {'supercategory:': 'mouse', 'id': 65, 'name': 'mouse'},
            {'supercategory:': 'remote', 'id': 66, 'name': 'remote'},
            {'supercategory:': 'keyboard', 'id': 67, 'name': 'keyboard'},
            {'supercategory:': 'cell phone', 'id': 68, 'name': 'cell phone'},
            {'supercategory:': 'microwave', 'id': 69, 'name': 'microwave'},
            {'supercategory:': 'oven', 'id': 70, 'name': 'oven'},
            {'supercategory:': 'toaster', 'id': 71, 'name': 'toaster'},
            {'supercategory:': 'sink', 'id': 72, 'name': 'sink'},
            {'supercategory:': 'refrigerator', 'id': 73, 'name': 'refrigerator'},
            {'supercategory:': 'book', 'id': 74, 'name': 'book'},
            {'supercategory:': 'clock', 'id': 75, 'name': 'clock'},
            {'supercategory:': 'vase', 'id': 76, 'name': 'vase'},
            {'supercategory:': 'scissors', 'id': 77, 'name': 'scissors'},
            {'supercategory:': 'teddy bear', 'id': 78, 'name': 'teddy bear'},
            {'supercategory:': 'hair\ndrier', 'id': 79, 'name': 'hair\ndrier'},
            {'supercategory:': 'toothbrush', 'id': 80, 'name': 'toothbrush'}]
        }
'''

trainpath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/ImageSets/'
datapath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages'
datapath2 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages2'
datapath3 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages_crossbar'
annopath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/Annotations/'
trainsetfile = 'train1030.txt'
outputpath = '/home/luyujie/SNIPER/data/coco'
phase = 'train'

classes = {'car': 1, 'van': 1, 'bus': 2, 'truck': 3, 'forklift': 3, 'person': 4, 'person-sitting': 4, 'bicycle': 5, 'motor': 5, 'open-tricycle': 6, 'close-tricycle': 6, 'water-block': 7, 'cone-block': 7, 'other-block': 7, 'crash-block': 7, 'triangle-block': 7, 'warning-block': 7, 'small-block': 7, 'large-block': 7,'bicycle-group': 20, 'person-group': 21, 'motor-group': 22, 'parked-bicycle': 23, 'parked-motor': 24, 'cross-bar': 25}
'''
classes = {db_info.classes = ['BG':1, 'person':2, 'bicycle':3, 'car':4, 'motorcycle':5, 'airplane':6,
                       'bus':7, 'train':8, 'truck':9, 'boat':10, 'traffic light':11, 'fire hydrant':12,
                       'stop sign':13, 'parking meter':14, 'bench':15, 'bird':16, 'cat':17, 'dog':18, 'horse':19, 'sheep':20, 'cow':21,
                       'elephant':22, 'bear':23, 'zebra':24, 'giraffe':25, 'backpack':26, 'umbrella':27, 'handbag':28, 'tie':29,
                       'suitcase':30, 'frisbee':31, 'skis':32, 'snowboard':33, 'sports\nball':34, 'kite':35, 'baseball\nbat':36,
                       'baseball glove':37, 'skateboard':38, 'surfboard':39, 'tennis racket':40, 'bottle':41, 'wine\nglass':42,
                       'cup':43, 'fork':44, 'knife':45, 'spoon':46, 'bowl':47, 'banana':48, 'apple':49, 'sandwich':50, 'orange':51,
                       'broccoli':52, 'carrot':53, 'hot dog':54, 'pizza':55, 'donut':56, 'cake':57, 'chair':58, 'couch':59,
                       'potted plant':60, 'bed':61, 'dining table':62, 'toilet':63, 'tv':64, 'laptop':65, 'mouse':66, 'remote':67,
                       'keyboard':68, 'cell phone':69, 'microwave':70, 'oven':71, 'toaster':72, 'sink':73, 'refrigerator':74, 'book':75,
                       'clock':76, 'vase':77, 'scissors':78, 'teddy bear':79, 'hair\ndrier':80, 'toothbrush':81]}
'''
with open(trainpath + trainsetfile) as f:
    count = 1
    cnt = 0
    annoid = 0
    for line in f:
        cnt += 1
        #if cnt > 1000:
        #    break
        #print line
        # line + .jpg
        imagepath = os.path.join(datapath, line.strip() + '.jpg')
        # no obstacle currently drop it
        txtpath = os.path.join(annopath, line.strip() + '.txt')
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

        #im = cv2.imread(imagepath)
            
        #height, width, _ = im.shape
        height = 1200
        width = 1920
        #print cnt
        if cnt % 1000 == 0:
            print cnt
        # resize
        #if height > 800 and width > 800:
            #print '>>> resize'
            #print imagepath
            #height = int(height * 2 / 3)
            #width = int(width * 2 / 3)
            #size = (width, height)
            #size = (int(width*2/3), int(height*2/3))
            #shrink = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            #cv2.imwrite(imagepath, shrink)
            #newpath = os.path.join(datapath, line.strip() + 'shrink' + '.jpg')
            #print newpath
            #cv2.imwrite(newpath, shrink)
            #cv2.waitKey(0)
        s = str(cnt).zfill(12)
        newimgpath = os.path.join(outputpath, 'images/train1030', 'COCO_train2014_' + s + '.jpg')
        #shutil.copy(imagepath, newimgpath)

        dataset['images'].append({'license': 5, 'file_name': newimgpath, 'coco_url': "local", 'height': height, 'width': width, 'date_captured': "2018_08_29 10:10:10", 'flickr_url': "local", 'id': cnt})
        #dataset['images'].append({'file_name': imagepath})
        # line + .txt
        #txtpath = os.path.join(annopath, line.strip() + '.txt')
        with open(txtpath) as annof:
            annos = annof.readlines()
        
        for ii, anno in enumerate(annos):
            parts = anno.strip().split()
            # resize bbox *2/3
            #x1
            x1 = float(parts[0])# * 2 / 3
            #y1
            y1 = float(parts[1])# * 2 / 3
            #x2 
            x2 = float(parts[2])# * 2 / 3
            #y2
            y2 = float(parts[3])# * 2 / 3
            #occlusion_level
            olevel = float(parts[4])
            #category
            if olevel > 1:
                continue
            category = parts[5]
            wid = max(0, x2 - x1)
            hei = max(0, y2 - y1)
            if hei < 20:
                continue
            if category.find("group") == -1:
                iscrowd = 0
            else:
                iscrowd = 1
            if classes[category] > 7:
                continue
            annoid = annoid + 1
                #print category
            #print category
            #print classes[category]
            '''
            x1 = x1 / 3
            y1 = y1 / 3
            wid = wid / 3
            hei = hei / 3
            '''
            dataset['annotations'].append({
                'segmentation': [],
                'iscrowd': iscrowd,
                'area': wid * hei,
                'image_id': cnt,
                'bbox': [x1, y1, wid, hei],
                'category_id': classes[category],
                'id': annoid
            })
        count += 1

folder = os.path.join(outputpath, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(outputpath, 'annotations/{}.json'.format(phase))

with open(json_name, 'w') as f:
    json.dump(dataset, f)
