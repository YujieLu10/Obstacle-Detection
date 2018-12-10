import init
import matplotlib
matplotlib.use('Agg')
from symbols.faster import *
from configs.faster.default_configs import config, update_config, update_config_from_list
from data_utils.load_data import load_proposal_roidb
import mxnet as mx
import argparse
from train_utils.utils import create_logger, load_param
from inference import imdb_detection_wrapper
from inference import imdb_proposal_extraction_wrapper
from dataset.pycocotools.cocoeval import COCOeval
import os
def main():
    print('>>>> cocoeval')
    COCOeval().evaluate()
    COCOeval().accumulate()
    COCOeval().summarize()

if __name__ == '__main__':
    main()
