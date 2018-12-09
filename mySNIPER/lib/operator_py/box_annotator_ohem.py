# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool




class BoxAnnotatorOHEMOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, num_reg_classes, roi_per_img,get_fg_labels=False):
        super(BoxAnnotatorOHEMOperator, self).__init__()
        self._num_classes = num_classes
        self._num_reg_classes = num_reg_classes
        self._roi_per_img = roi_per_img
        self.get_fg_labels = get_fg_labels

    def forward(self, is_train, req, in_data, out_data, aux):
        
        cls_score    = in_data[0]
        bbox_pred    = in_data[1]
        labels       = in_data[2].asnumpy()
        bbox_targets = in_data[3]
        bbox_weights = in_data[4]
        labels_ohem = labels.copy()
        bbox_weights_ohem = bbox_weights.asnumpy().copy()
        batch_size = np.shape(labels)[0]
        num_rois = np.shape(labels)[1]
        for i in range(batch_size):
            per_roi_loss_cls = mx.nd.SoftmaxActivation(cls_score[i]) + 1e-14
            per_roi_loss_cls = per_roi_loss_cls.asnumpy()
            num_rois = np.shape(labels[i])[0]
            vids = np.where(labels[i] >= 0)[0]
            nvids = np.where(labels[i] < 0)[0]
            labels[i][nvids] = 0

            per_roi_loss_cls = per_roi_loss_cls[np.arange(per_roi_loss_cls.shape[0], dtype='int'), labels[i].astype('int')]
            per_roi_loss_cls = -1 * np.log(per_roi_loss_cls)
            per_roi_loss_cls = np.reshape(per_roi_loss_cls, newshape=(-1,))
            per_roi_loss_cls[nvids] = 0

            per_roi_loss_bbox = bbox_weights[i] * mx.nd.smooth_l1((bbox_pred[i] - bbox_targets[i]), scalar=1.0)
            per_roi_loss_bbox = mx.nd.sum(per_roi_loss_bbox, axis=1).asnumpy()
            per_roi_loss_bbox[nvids] = 0

            top_k_per_roi_loss = np.argsort(per_roi_loss_cls + per_roi_loss_bbox)
            labels[i][nvids] = -1
            labels_ohem[i] = labels[i]
            labels_ohem[i][top_k_per_roi_loss[::-1][self._roi_per_img:]] = -1
            bbox_weights_ohem[i] = bbox_weights[i].asnumpy()
            bbox_weights_ohem[i][top_k_per_roi_loss[::-1][self._roi_per_img:]] = 0

        

        if not self.get_fg_labels:
            labels_ohem = mx.nd.array(labels_ohem)
            bbox_weights_ohem = mx.nd.array(bbox_weights_ohem)
            for ind, val in enumerate([labels_ohem, bbox_weights_ohem]):
                self.assign(out_data[ind], req[ind], val)
        else:
            fg_labels = labels_ohem.copy()
            for i in range(batch_size):
                bg_inds = np.where(labels_ohem[i]==0)[0]
                fg_labels[i][bg_inds] = -1
            labels_ohem = mx.nd.array(labels_ohem)
            bbox_weights_ohem = mx.nd.array(bbox_weights_ohem)
            fg_labels = mx.nd.array(fg_labels)
            for ind, val in enumerate([labels_ohem, bbox_weights_ohem,fg_labels]):
                self.assign(out_data[ind], req[ind], val)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('BoxAnnotatorOHEM')
class BoxAnnotatorOHEMProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, num_reg_classes, roi_per_img,get_fg_labels=False):
        super(BoxAnnotatorOHEMProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._num_reg_classes = int(num_reg_classes)
        self._roi_per_img = int(roi_per_img)
        self.get_fg_labels = get_fg_labels
    def list_arguments(self):
        return ['cls_score', 'bbox_pred', 'labels', 'bbox_targets', 'bbox_weights']

    def list_outputs(self):
        if not self.get_fg_labels:
            return ['labels_ohem', 'bbox_weights_ohem']
        else:
            return ['labels_ohem', 'bbox_weights_ohem','fg_labels_ohem']

    def infer_shape(self, in_shape):
        labels_shape = in_shape[2]
        bbox_weights_shape = in_shape[4]
        #print "quack"
        #print labels_shape
        #print in_shape
        if not self.get_fg_labels:
            return in_shape, \
               [labels_shape, bbox_weights_shape]
        else:
            return in_shape, \
               [labels_shape, bbox_weights_shape, labels_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BoxAnnotatorOHEMOperator(self._num_classes, self._num_reg_classes, self._roi_per_img,self.get_fg_labels)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
