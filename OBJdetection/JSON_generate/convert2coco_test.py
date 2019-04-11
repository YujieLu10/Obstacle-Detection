import sys
import json
import cv2
import os
import shutil
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
            {'supercategory:': 'motor', 'id': 4, 'name': 'motor'},
            {'supercategory:': 'airplane', 'id': 5, 'name': 'airplane'},
            {'supercategory:': 'bus', 'id': 6, 'name': 'bus'},
            {'supercategory:': 'van', 'id': 7, 'name': 'van'},
            {'supercategory:': 'truck', 'id': 8, 'name': 'truck'},
            {'supercategory:': 'forklift', 'id': 9, 'name': 'forklift'},
            {'supercategory:': 'traffic light', 'id': 10, 'name': 'traffic light'},
            {'supercategory:': 'fire hydrant', 'id': 11, 'name': 'fire hydrant'},
            {'supercategory:': 'stop sign', 'id': 12, 'name': 'stop sign'},
            {'supercategory:': 'person sitting', 'id': 13, 'name': 'person sitting'},
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
            {'supercategory:': 'open-tricycle', 'id': 65, 'name': 'open-tricycle'},
            {'supercategory:': 'close-tricycle', 'id': 66, 'name': 'close-tricycle'},
            {'supercategory:': 'water-block', 'id': 67, 'name': 'water-block'},
            {'supercategory:': 'cone-block', 'id': 68, 'name': 'cone-block'},
            {'supercategory:': 'other-block', 'id': 69, 'name': 'other-block'},
            {'supercategory:': 'crash-block', 'id': 70, 'name': 'crash-block'},
            {'supercategory:': 'triangle-block', 'id': 71, 'name': 'triangle-block'},
            {'supercategory:': 'warning-block', 'id': 72, 'name': 'warning-block'},
            {'supercategory:': 'small-block', 'id': 73, 'name': 'small-block'},
            {'supercategory:': 'large-block', 'id': 74, 'name': 'large-block'},
            {'supercategory:': 'bicycle-group', 'id': 75, 'name': 'bicycle-group'},
            {'supercategory:': 'person-group', 'id': 76, 'name': 'person-group'},
            {'supercategory:': 'motor-group', 'id': 77, 'name': 'motor-group'},
            {'supercategory:': 'parked-bicycle', 'id': 78, 'name': 'parked-bicycle'},
            {'supercategory:': 'parked-motor', 'id': 79, 'name': 'parked-motor'},
            {'supercategory:': 'cross-bar', 'id': 80, 'name': 'cross-bar'}]
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
            {'supercategory:': 'car', 'id': 1, 'name': 'car'},
            {'supercategory:': 'bus', 'id': 2, 'name': 'bus'},
            {'supercategory:': 'truck', 'id': 3, 'name': 'truck'},
            {'supercategory:': 'person', 'id': 4, 'name': 'person'},
            {'supercategory:': 'bicycle', 'id': 5, 'name': 'bicycle'},
            {'supercategory:': 'tricycle', 'id': 6, 'name': 'tricycle'},
            {'supercategory:': 'block', 'id': 7, 'name': 'block'}]
        }
testpath = '/private/luyujie/obstacle_detector/obstacle_detector/data/obstacle2d/ImageSets/'
datapath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages'
datapath2 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages2'
datapath3 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages_crossbar'
annopath = '/private/luyujie/obstacle_detector/obstacle_detector/data/obstacle2d/Annotations/'
testsetfile = 'val1010.txt'
outputpath = '/home/luyujie/SNIPER/data/coco'
phase = 'test'

classes = {'car': 1, 'van': 1, 'bus': 2, 'truck': 3, 'forklift': 3, 'person': 4, 'person-sitting': 4, 'bicycle': 5, 'motor': 5, 'open-tricycle': 6, 'close-tricycle': 6, 'water-block': 7, 'cone-block': 7, 'other-block': 7, 'crash-block': 7, 'triangle-block': 7, 'warning-block': 7, 'small-block': 7, 'large-block': 7,'bicycle-group': 20, 'person-group': 21, 'motor-group': 22, 'parked-bicycle': 23, 'parked-motor': 24, 'cross-bar': 25}
'''

classes = {'BG':1, 'person':2, 'bicycle':3, 'car':4, 'motor':5, 'airplane':6,
                       'bus':7, 'van':8, 'truck':9, 'forklift':10, 'traffic light':11, 'fire hydrant':12,
                       'stop sign':13, 'person-sitting':2, 'bench':15, 'bird':16, 'cat':17, 'dog':18, 'horse':19, 'sheep':20, 'cow':21,
                       'elephant':22, 'bear':23, 'zebra':24, 'giraffe':25, 'backpack':26, 'umbrella':27, 'handbag':28, 'tie':29,
                       'suitcase':30, 'frisbee':31, 'skis':32, 'snowboard':33, 'sports\nball':34, 'kite':35, 'baseball\nbat':36,
                       'baseball glove':37, 'skateboard':38, 'surfboard':39, 'tennis racket':40, 'bottle':41, 'wine\nglass':42,
                       'cup':43, 'fork':44, 'knife':45, 'spoon':46, 'bowl':47, 'banana':48, 'apple':49, 'sandwich':50, 'orange':51,
                       'broccoli':52, 'carrot':53, 'hot dog':54, 'pizza':55, 'donut':56, 'cake':57, 'chair':58, 'couch':59,
                       'potted plant':60, 'bed':61, 'dining table':62, 'toilet':63, 'tv':64, 'laptop':65, 'open-tricycle':66, 'close-tricycle':67,
                       'water-block':68, 'cone-block':69, 'other-block':70, 'crash-block':71, 'triangle-block':72, 'warning-block':73, 'small-block':74, 'large-block':75,
                       'bicycle-group':76, 'person-group':2, 'motor-group':78, 'parked-bicycle':79, 'parked-motor':80, 'cross-bar':81}
'''
with open(testpath + testsetfile) as f:
    count = 1
    cnt = 0
    annoid = 0
    for line in f:
        cnt += 1
        #small
        #if cnt > 100:
        #    break
        if cnt % 1000 == 0:
            print cnt
        #print line
        # line + .jpg
        imagepath = os.path.join(datapath, line.strip() + '.jpg')
        # no obstacle currently drop it
        txtpath = os.path.join(annopath, line.strip() + '.txt')
        if not os.path.exists(txtpath):
            print txtpath
            continue
        if not os.path.exists(imagepath):
            imagepath = os.path.join(datapath2, line.strip() + '.jpg')
            if not os.path.exists(imagepath):
                imagepath = os.path.join(datapath3, line.strip() + '.jpg')
        if not os.path.exists(imagepath):
            continue
        #print imagepath
        #im = cv2.imread(imagepath)
        #height, width, _ = im.shape
        height = 1200
        width = 1920
        #if height > 800 and width > 800: 
            #height = int(height * 2 / 3)
            #width = int(width * 2 / 3)
        #size = (width, height)
        #shrink = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        #cv2.imwrite(imagepath, shrink)
        s = str(cnt).zfill(12)
        newimgpath = os.path.join(outputpath, 'images/test2015', 'COCO_test2015_' + s + '.jpg')
        shutil.copy(imagepath, newimgpath)
        dataset['images'].append({'license': 5, 'file_name': newimgpath, 'coco_url': "local", 'height': height, 'width': width, 'date_captured': "2018_08_29 10:10:10", 'flickr_url': "local", 'id': cnt})
        #dataset['images'].append({'file_name': imagepath})
        # line + .txt
        #txtpath = os.path.join(annopath, line.strip() + '.txt')
        
        with open(txtpath) as annof:
            annos = annof.readlines()
        
        for ii, anno in enumerate(annos):
            parts = anno.strip().split()

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
            #print parts[5]
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
            '''
            x1 = x1 / 3
            y1 = y1 / 3
            wid = wid / 3
            hei = hei / 3
            '''
            dataset['annotations'].append({
                'segmentation':[], #[x1, y2, x2, y2, x2, y1, x1, y1],
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
