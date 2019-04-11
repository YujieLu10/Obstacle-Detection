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
trainpath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/ImageSets/'
datapath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages'
datapath2 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages2'
datapath3 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages_crossbar'
annopath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/Annotations/'
trainsetfile = 'train0829.txt'
outputpath = '/home/luyujie/SNIPER/data/coco'
phase = 'train'
classes = {'car': 1, 'van': 1, 'bus': 2, 'truck': 3, 'forklift': 3, 'person': 4, 'person-sitting': 4, 'bicycle': 5, 'motor': 5, 'open-tricycle': 6, 'close-tricycle': 6, 'water-block': 7, 'cone-block': 7, 'other-block': 7, 'crash-block': 7, 'triangle-block': 7, 'warning-block': 7, 'small-block': 7, 'large-block': 7,'bicycle-group': 20, 'person-group': 21, 'motor-group': 22, 'parked-bicycle': 23, 'parked-motor': 24, 'cross-bar': 25}
with open(trainpath + trainsetfile) as f:
    count = 0
    cnt = 0
    annoid = 0
    for line in f:
        cnt += 1
        #if cnt > 1000:
        #    break
        # line + .jpg
        imagepath = os.path.join(datapath, line.strip() + '.jpg')
        # no obstacle currently drop it
        txtpath = os.path.join(annopath, line.strip() + '.txt')
        if not os.path.exists(txtpath):
            print txtpath
            continue
        #print imagepath
        #print imagepath
        if not os.path.exists(imagepath):
            imagepath = os.path.join(datapath2, line.strip() + '.jpg')
            if not os.path.exists(imagepath):
                imagepath = os.path.join(datapath3, line.strip() + '.jpg')
        if not os.path.exists(imagepath):
            print imagepath
            continue
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
        newimgpath = os.path.join(outputpath, 'images/train2014', 'COCO_train2014_' + s + '.jpg')
        #shutil.copy(imagepath, newimgpath)
        dataset['images'].append({'license': 5, 'file_name': newimgpath, 'coco_url': "local", 'height': height, 'width': width, 'date_captured': "2018_08_29 10:10:10", 'flickr_url': "local", 'id': cnt})
        #dataset['images'].append({'file_name': imagepath})
        # line + .txt
        #txtpath = os.path.join(annopath, line.strip() + '.txt')
        # if no txtpath won't add detected person annotation as well
        addtxtpath = os.path.join('/home/luyujie/imid_anno', str(count) + '.txt')
        with open(addtxtpath) as addannof:
            annos = addannof.readlines()
            #print addtxtpath
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
            if olevel > 1:
                continue
            category = parts[5]
            wid = max(0, x2 - x1)
            hei = max(0, y2 - y1)
            if category.find("group") == -1:
                iscrowd = 0
            else:
                iscrowd = 1
            if hei < 30:
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
                'image_id': cnt,
                'area': wid * hei,
                'bbox': [x1, y1, wid, hei],
                'category_id': classes[category],
                'id': annoid
            })


        with open(txtpath) as annof:
            annos = annof.readlines()      
            #print txtpath  
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
            if olevel > 1:
                continue
            category = parts[5]
            wid = max(0, x2 - x1)
            hei = max(0, y2 - y1)
            if category.find("group") == -1:
                iscrowd = 0
            else:
                iscrowd = 1
            # ignore annotated person

            if classes[category] == 4:
                continue
            if classes[category] > 7:
                #print category
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
                'image_id': cnt,
                'area': wid * hei,
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
