# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER demo
# by Mahyar Najibi
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from configs.faster.default_configs import config, update_config, update_config_from_list
import mxnet as mx
import argparse
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference_txt import Tester
from symbols.faster import *
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


trainpath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/ImageSets/'
datapath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages'
datapath2 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages2'
datapath3 = '/private/ningqingqun/obstacle_detector/data/obstacle2d/JPGImages_crossbar'
annopath = '/private/ningqingqun/obstacle_detector/data/obstacle2d/Annotations/'
annoaddpath = '/home/luyujie/addAnnotations/'
trainsetfile = 'train0829.txt'
outputpath = '/home/luyujie/SNIPER/data/coco'
phase = 'addAnnoTxt'
'''
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



def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--im_path', dest='im_path', help='Path to the image', type=str,
                            default='data/demo/demo.jpg')
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()


def main():
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    # Use just the first GPU for demo
    context = [mx.gpu(int(1))]
    #context = [mx.gpu(int(gpu)) for gpu in config.gpus.split(',')]
    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    with open(trainpath + trainsetfile) as f:
        count = 1
        cnt = 0
        annoid = 0
        for line in f:
            cnt += 1
            #if cnt > 1000:
            #    break
            #print line
            addtxtpath = os.path.join(annoaddpath, line.strip() + '_person' + '.txt')
            if os.path.exists(addtxtpath):
                print addtxtpath
                continue
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
            #width, height = Image.open(imagepath).size

            # Pack image info
            roidb = [{'image': imagepath, 'width': width, 'height': height, 'flipped': False}]

            # Creating the Logger
            logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

            # Pack db info
            db_info = EasyDict()
            db_info.name = 'coco'
            db_info.result_path = 'data/demo'

            # Categories the detector trained for:
            
            db_info.classes = [u'BG', u'person', u'bicycle', u'car', u'motorcycle', u'airplane',
                        u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant',
                        u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow',
                        u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie',
                        u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports\nball', u'kite', u'baseball\nbat',
                        u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine\nglass',
                        u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange',
                        u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch',
                        u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote',
                        u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book',
                        u'clock', u'vase', u'scissors', u'teddy bear', u'hair\ndrier', u'toothbrush']
            '''
            db_info.classes = [u'BG', u'car', u'bus', u'truck', u'person', u'bicycle', u'tricycle', u'block']
            '''
            db_info.num_classes = len(db_info.classes)

            # Create the model
            sym_def = eval('{}.{}'.format(config.symbol, config.symbol))
            sym_inst = sym_def(n_proposals=400, test_nbatch=1)
            sym = sym_inst.get_symbol_rcnn(config, is_train=False)
            test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=1, nGPUs=1, threads=1,
                                    crop_size=None, test_scale=config.TEST.SCALES[0],
                                    num_classes=db_info.num_classes)
            # Create the module
            shape_dict = dict(test_iter.provide_data_single)
            sym_inst.infer_shape(shape_dict)
            mod = mx.mod.Module(symbol=sym,
                                context=context,
                                data_names=[k[0] for k in test_iter.provide_data_single],
                                label_names=None)
            mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)

            # Initialize the weights
            model_prefix = os.path.join(output_path, args.save_prefix)
            arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                                convert=True, process=True)
            mod.init_params(arg_params=arg_params, aux_params=aux_params)

            # Create the tester
            tester = Tester(mod, db_info, roidb, test_iter, cfg=config, batch_size=1)

            # Sequentially do detection over scales
            # NOTE: if you want to perform detection on multiple images consider using main_test which is parallel and faster
            all_detections= []
            for s in config.TEST.SCALES:
                # Set tester scale
                tester.set_scale(s)
                # Perform detection
                all_detections.append(tester.get_detections(vis=False, evaluate=False, cache_name=None))

            # Aggregate results from multiple scales and perform NMS
            tester = Tester(None, db_info, roidb, None, cfg=config, batch_size=1)
            file_name, out_extension = os.path.splitext(os.path.basename(imagepath))
            #print('>>> all detections {}'.format(all_detections))
            last_position = -1
            while True:
                position = line.find("/", last_position + 1)
                if position == -1:
                    break
                last_position = position
            dirpath = os.path.join(annoaddpath, line.strip()[0:last_position])
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            all_detections = tester.aggregate(all_detections, vis=True, cache_name=None, vis_path='/home/luyujie/addAnno_txt_vis',
                                                vis_name='{}'.format(file_name), vis_ext=out_extension, addtxtpath = os.path.join(annoaddpath, line.strip() + '_person' + '.txt'))

            s = str(cnt).zfill(12)
            newimgpath = os.path.join(outputpath, 'images/train2014', 'COCO_train2014_' + s + '.jpg')
            #shutil.copy(imagepath, newimgpath)

    return all_detections

if __name__ == '__main__':
    main()
