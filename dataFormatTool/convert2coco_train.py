import sys
import json
import cv2
import os

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
            {'supercategory:': 'car', 'id': 2, 'name': 'van'},
            {'supercategory:': 'bus', 'id': 3, 'name': 'bus'},
            {'supercategory:': 'truck', 'id': 4, 'name': 'truck'},
            {'supercategory:': 'truck', 'id': 5, 'name': 'forklift'},
            {'supercategory:': 'person', 'id': 6, 'name': 'person'},
            {'supercategory:': 'person', 'id': 7, 'name': 'person-sitting'},
            {'supercategory:': 'bicycle', 'id': 8, 'name': 'bicycle'},
            {'supercategory:': 'bicycle', 'id': 9, 'name': 'motor'},
            {'supercategory:': 'tricycle', 'id': 10, 'name': 'open-tricycle'},
            {'supercategory:': 'tricycle', 'id': 11, 'name': 'close-tricycle'},
            {'supercategory:': 'block', 'id': 12, 'name': 'water-block'},
            {'supercategory:': 'block', 'id': 13, 'name': 'cone-block'},
            {'supercategory:': 'block', 'id': 14, 'name': 'other-block'},
            {'supercategory:': 'block', 'id': 15, 'name': 'crash-block'},
            {'supercategory:': 'block', 'id': 16, 'name': 'triangle-block'},
            {'supercategory:': 'block', 'id': 17, 'name': 'warning-block'},
            {'supercategory:': 'block', 'id': 18, 'name': 'small-block'},
            {'supercategory:': 'block', 'id': 19, 'name': 'large-block'},
            {'supercategory:': 'ignore_classes', 'id': 20, 'name': 'bicycle-group'},
            {'supercategory:': 'ignore_classes', 'id': 21, 'name': 'person-group'},
            {'supercategory:': 'ignore_classes', 'id': 22, 'name': 'motor-group'},
            {'supercategory:': 'ignore_classes', 'id': 23, 'name': 'parked-bicycle'},
            {'supercategory:': 'ignore_classes', 'id': 24, 'name': 'parked-motor'},
            {'supercategory:': 'ignore_classes', 'id': 25, 'name': 'cross-bar'},
            {'supercategory:': 'ignore_classes', 'id': 26, 'name': 'bicycle-group'},
            {'supercategory:': 'ignore_classes', 'id': 27, 'name': 'person-group'},
            {'supercategory:': 'ignore_classes', 'id': 28, 'name': 'motor-group'},
            {'supercategory:': 'ignore_classes', 'id': 29, 'name': 'parked-bicycle'},
            {'supercategory:': 'ignore_classes', 'id': 30, 'name': 'parked-motor'},
            {'supercategory:': 'ignore_classes', 'id': 31, 'name': 'cross-bar'}]
        }
trainpath = '/private/luyujie/obstacle_detector/obstacle_detector/data/obstacle2d/ImageSets/'
datapath = '/private/luyujie/obstacle_detector/obstacle_detector/data/obstacle2d/JPGImages/'
annopath = '/private/luyujie/obstacle_detector/obstacle_detector/data/obstacle2d/Annotations/'
trainsetfile = 'train0829.txt'
outputpath = '/home/luyujie/SNIPER/data/coco'
phase = 'train'
classes = {'car': 1, 'van': 2, 'bus': 3, 'truck': 4, 'forklift': 5, 'person': 6, 'person-sitting': 7, 'bicycle': 8, 'motor': 9, 'open-tricycle': 10, 'close-tricycle': 11, 'water-block': 12, 'cone-block': 13, 'other-block': 14, 'crash-block': 15, 'triangle-block': 16, 'warning-block': 17, 'small-block': 18, 'large-block': 19, 'bicycle-group': 20, 'person-group': 21, 'motor-group': 22, 'parked-bicycle': 23, 'parked-motor': 24, 'cross-bar': 25,'bicycle-group': 26, 'person-group': 27, 'motor-group': 28,'parked-bicycle': 29, 'parked-motor': 30, 'cross-bar': 31}
with open(trainpath + trainsetfile) as f:
    count = 1
    for line in f:
        #print line
        # line + .jpg
        imagepath = os.path.join(datapath, line.strip() + '.jpg')
        # no obstacle currently drop it
        txtpath = os.path.join(annopath, line.strip() + '.txt')
        if not os.path.exists(txtpath):
            print txtpath
            continue
        #print imagepath
        im = cv2.imread(imagepath)
        height, width, _ = im.shape

        dataset['images'].append({'license': 5, 'file_name':  imagepath, 'coco_url': "local", 'height': height, 'width': width, 'date_captured': "2018_08_29 10:10:10", 'flickr_url': "local", 'id': count})
        #dataset['images'].append({'file_name': imagepath})
        # line + .txt
        #txtpath = os.path.join(annopath, line.strip() + '.txt')
        with open(txtpath) as annof:
            annos = annof.readlines()
        
        for ii, anno in enumerate(annos):
            parts = anno.strip().split()

            #x1
            x1 = float(parts[0])
            #y1
            y1 = float(parts[1])
            #x2 
            x2 = float(parts[2])
            #y2
            y2 = float(parts[3])
            #occlusion_level
            olevel = float(parts[4])
            #category
            category = parts[5]
            wid = max(0, x2 - x1)
            hei = max(0, y2 - y1)
            dataset['annotations'].append({
                'segmentation': [x1, y2, x2, y2, x2, y1, x1, y1],
                'area': wid * hei,
                'iscrowd': 0,
                'image_id': count,
                'bbox': [x1, y1, wid, hei],
                'category_id': classes[category],
                'id': ii
            })
        count += 1

folder = os.path.join(outputpath, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(outputpath, 'annotations/{}.json'.format(phase))

with open(json_name, 'w') as f:
    json.dump(dataset, f)
