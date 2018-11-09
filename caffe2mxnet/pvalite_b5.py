import mxnet as mx
from symbols.symbol import Symbol
from operator_py.box_annotator_ohem import *
import numpy as np

def checkpoint_callback(bbox_param_names, prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg[bbox_param_names[0]]
        bias = arg[bbox_param_names[1]]
        stds = np.array([0.1, 0.1, 0.2, 0.2])
        arg[bbox_param_names[0] + '_test'] = (weight.T * mx.nd.array(stds)).T
        arg[bbox_param_names[1] + '_test'] = bias * mx.nd.array(stds)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop(bbox_param_names[0] + '_test')
        arg.pop(bbox_param_names[1] + '_test')
    return _callback

class pvalite_b5(Symbol):
    def __init__(self, n_proposals=400, momentum=0.95, fix_bn=False, test_nbatch=1):
        self.workspace = 512
        self.momentum = momentum
        self.fix_bn = fix_bn
        self.test_nbatch= test_nbatch

    def get_bbox_param_names(self):
        return ['bbox_pred_weight', 'bbox_pred_bias']

    def inc3_unit_left(self, data, name, workspace=512):
        conv5_1 = mx.sym.Convolution(data=data,num_filter=16,kernel=(1,1),stride=(1,1),pad=0,no_bias=False,workspace=workspace,name=name+'/conv5_1')
        relu5_1 = mx.sym.Activation(data=conv5_1, act_type='relu', name=name+'/relu5_1')
        if 'inc4' in name or 'inc5' in name:
            conv5_2 = mx.sym.Convolution(data=relu5_1, num_filter=32,kernel=(3,3),stride=(1,1),dilate=(2,2),pad=(1,1),no_bias=False,workspace=workspace,name=name+'/conv5_2')
            relu5_2 = mx.sym.Activation(data=conv5_2, act_type='relu', name=name+'/relu5_2')
            conv5_3 = mx.sym.Convolution(data=relu5_2, num_filter=32,kernel=(3,3),stride=(1,1),dilate=(2,2),pad=(1,1),no_bias=False,workspace=workspace,name=name+'/conv5_3')
            relu5_3 = mx.sym.Activation(data=conv5_3, act_type='relu', name=name+'/relu5_3')
        else:
            conv5_2 = mx.sym.Convolution(data=relu5_1, num_filter=32,kernel=(3,3),stride=(1,1),pad=(1,1),no_bias=False,workspace=workspace,name=name+'/conv5_2')
            relu5_2 = mx.sym.Activation(data=conv5_2, act_type='relu', name=name+'/relu5_2')
            conv5_3 = mx.sym.Convolution(data=relu5_2, num_filter=32,kernel=(3,3),stride=(1,1),pad=(1,1),no_bias=False,workspace=workspace,name=name+'/conv5_3')
            relu5_3 = mx.sym.Activation(data=conv5_3, act_type='relu', name=name+'/relu5_3')
        return relu5_3

    def inc3_unit_middle(self, data, name, workspace=512):
        conv3_1 = mx.sym.Convolution(data=data, num_filter=16,kernel=(1,1),stride=(1,1),pad=0,no_bias=False,workspace=workspace,name=name+'/conv3_1')
        relu3_1 = mx.sym.Activation(data=conv3_1,act_type='relu', name=name+'/relu3_1')

        if name == 'inc3a':
            incstride = (2,2)
        else:
            incstride = (1,1)
        if 'inc4' in name or 'inc5' in name:
            conv3_2 = mx.sym.Convolution(data=relu3_1, num_filter=64,kernel=(3,3),stride=incstride,dilate=(2,2),pad=(1,1),no_bias=False,workspace=workspace,name=name+'/conv3_2')
            relu3_2 = mx.sym.Activation(data=conv3_2,act_type='relu', name=name+'/relu3_2')
        else:        
            conv3_2 = mx.sym.Convolution(data=relu3_1, num_filter=64,kernel=(3,3),stride=incstride,pad=(1,1),no_bias=False,workspace=workspace,name=name+'/conv3_2')
            relu3_2 = mx.sym.Activation(data=conv3_2,act_type='relu', name=name+'/relu3_2')
        return relu3_2

    def inc3_unit_right(self, data, num_output, name, workspace=512):
        conv1 = mx.sym.Convolution(data=data, num_filter=num_output,kernel=(1,1),stride=(1,1),pad=(0,0),no_bias=False,workspace=workspace,name=name+'/conv1')
        relu1 = mx.sym.Activation(data=conv1, act_type='relu', name=name+'/relu1')
        return relu1

    def get_symbol_rcnn(self, cfg, is_train=True):
        workspace = self.workspace
        num_stage = 3
        char_stage = 5
        stage_num = ['3', '4', '5']
        stage_char = ['a', 'b', 'c', 'd', 'e']
        filter_list = [96, 128, 128]
        data = mx.sym.Variable(name="data")
        #label??
        rpn_label = mx.sym.Variable(name='label')
        im_info = mx.sym.Variable(name='im_info')
        rpn_bbox_weight = mx.sym.Variable('bbox_weight')
        rpn_bbox_target = mx.sym.Variable(name='bbox_target')
        im_ids = mx.sym.Variable(name='im_ids')

        conv1 = mx.sym.Convolution(data=data, num_filter = 32, kernel=(4, 4), stride=(2, 2), pad=(1, 1), no_bias=False, workspace=workspace, name='conv1')

        relu1 = mx.sym.Activation(data=conv1, act_type='relu', name='relu1')
        
        conv2 = mx.sym.Convolution(data=relu1, num_filter = 48, kernel=(3, 3),
        stride=(2, 2), pad=(1, 1), no_bias=False, workspace=workspace, name='conv2')

        relu2 = mx.sym.Activation(data=conv2, act_type='relu', name='relu2')

        conv3 = mx.sym.Convolution(data=relu2, num_filter = 96, kernel=(3, 3),
        stride=(2, 2), pad=(1, 1), no_bias=False, workspace=workspace, name='conv3')

        relu3 = mx.sym.Activation(data=conv3, act_type='relu', name='relu3')

        inc3_left = conv3
        inc3_middle = relu3
        inc3_right = conv3
        conv3_down2 = mx.symbol.Pooling(data=conv3, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max')
        #incleft
        for i in range (num_stage):
            print(stage_num[i])
            #inc3/pool
            if i == 0:
                inc3_right = mx.symbol.Pooling(data=inc3_right, kernel=(2 if i == 0 else 3, 2 if i == 0 else 3), stride=(2 if i == 0 else 1, 2 if i == 0 else 1), pad=(0 if i == 0 else 1, 0 if i == 0 else 1), pool_type='max')
            else:
                inc3_right = mx.symbol.Pooling(data=inc_concat, kernel=(2 if i == 0 else 3, 2 if i == 0 else 3), stride=(2 if i == 0 else 1, 2 if i == 0 else 1), pad=(0 if i == 0 else 1, 0 if i == 0 else 1), pool_type='max')
            for j in range (char_stage):
                print(stage_char[j])
                inc3_left = self.inc3_unit_left(inc3_left if i == 0 else inc_concat, 'inc' + stage_num[i] + stage_char[j], workspace)
                inc3_middle = self.inc3_unit_middle(inc3_middle if i == 0 else inc_concat, 'inc' + stage_num[i] + stage_char[j], workspace)
                inc3_right = self.inc3_unit_right(inc3_right if i == 0 else inc_concat, filter_list[i], 'inc' + stage_num[i] + stage_char[j], workspace)
                inc_concat = mx.sym.concat(inc3_left, inc3_middle, inc3_right, dim=3)
                #inc_concat = mx.sym.concat()
                if stage_char == 'e' and stage_num == '3':
                    inc3e = inc_concat
                
        #concat
        concat = mx.sym.concat(conv3_down2, inc_concat, inc3e)

        #convf/reluf
        convf = mx.sym.FullyConnected(name='convf', data = concat, lr_mult={0.1, 0.2})
        reluf = mx.sym.Activation(data = convf, act_type='relu', name='reluf')

        #grad_scale
        if cfg.TRAIN.fp16 == True:
            grad_scale = float(cfg.TRAIN.scale)
        else:
            grad_scale = 1.0

        rpn_conv1 = mx.sym.Convolution(data = convf, kernel=(1,1),pad=(0,0),num_filter=256,name='rpn_conv1')
        rpn_relu1 = mx.sym.Activation(data = rpn_conv1, act_type='relu', name='rpn_relu1')
        rpn_cls_score_fabu = mx.sym.Convolution(data = rpn_relu1, kernel=(1,1),pad=(0,0),num_filter=98,name='rpn_conv1')
        rpn_cls_score_reshape = mx.sym.Reshape(data = rpn_cls_score_fabu, shape=(0,2,-1,0),name='rpn_cls_score_reshape')
        rpn_bbox_pred_fabu = mx.sym.Convolution(data=rpn_relu1, kernel=(1,1),pad=(0,0),num_filter=196,name='rpn_bbox_pred_fabu')

        # generate anchor ?
        rpn_data = mx.nd.contrib.MultiBoxPrior(data = data, sizes=[1.5, 3, 6, 9, 16, 32, 48],ratios=[0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0],steps=[16,16],name='rpn_data')
        #rpn_loss_bbox
        rpn_loss_bbox = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_loss_bbox',scalar=1.0,data=(rpn_bbox_pred_fabu - rpn_bbox_target))
        rpn_loss_bbox = mx.sym.MakeLoss(name='rpn_loss_bbox', data=rpn_loss_bbox,grad_scale=3*grad_scale / float(cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))

        rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True, normalization='valid',use_ignore=True, ignore_label=-1,name='rpn_cls_prob',grad_scale=grad_scale)
        rpn_cls_prob_reshape = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 98, -1, 0),name='rpn_cls_prob_reshape')
        
        #batchsize?? self.test_nbatch
        proposal, rpn_scores = mx.sym.MultiProposal(cls_prob=rpn_cls_prob_reshape,bbox_pred=rpn_bbox_pred_fabu, im_info=im_info,name='proposal', batch_size=16,
        rpn_pre_nms_top_n=10000,rpn_post_nms_top_n=2000, rpn_min_size=8, threshold=0.7,feature_stride=16,ratios=(0.333, 0.5, 0.667, 1, 1.5, 2, 3),scales=(1.5, 3, 6, 9, 16, 32, 48))

        #rpn_loss_cls =
        #mute_rpn_scores = ?
        if is_train:
            group = mx.sym.Group([rpn_cls_prob, rpn_loss_bbox])
        else:
            group = mx.sym.Group([proposal, rpn_scores, im_ids])
        self.sym = group


        #roi_data python?
        #roi_data = 
        roi_pool_conv5 = mx.sym.ROIPooling(data = data,rois = roi_data, pooled_size = (6, 6), spatial_scale=0.0625, name = 'roi_pool_conv5')

        #fc6_L = mx.sym.FullyConnected(name='fc6_L', data=roi_pool_conv5)
        fc6_L = mx.sym.CaffeOp(data_0=roi_pool_conv5, prototxt="layer {type:\"InnerProduct\" inner_product_param {num_output: 512}}")
        #fc6_U = mx.sym.FullyConnected(name = 'fc6_U', data = fc6_L)
        fc6_U = mx.sym.CaffeOp(data_0=fc6_L, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
        relu6 = mx.sym.Activation(data = fc6_U, act_type = 'relu', name = 'relu6')

        #fc7_L = mx.sym.FullyConnected(name = 'fc7_L', data = relu6)
        fc7_L = mx.sym.CaffeOp(data_0=relu6, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 128}}")
        #fc7_U = mx.sym.FullyConnected(name = 'fc7_U', data = fc7_L)
        fc7_U = mx.sym.CaffeOp(data_0=fc7_L, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
        relu7 = mx.sym.FullyConnected(name = 'relu7', data = fc7_U)

        cls_score_fabu = mx.sym.CaffeOp(data_0=fc7_U, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 8}}")
        bbox_pred_fabu = mx.sym.CaffeOp(data_0=relu7, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 32}}")
        #loss_cls = softmaxwithloss
        #label -> labels -> roi-data/labels
        loss_cls = mx.sym.CaffeLoss(data = cls_score_fabu, label = label, grad_scale = 1, name='loss_cls', prototxt="layer{type:\"SoftmaxWithLoss\"}")
        #num_weight=?
        #loss_bbox = ? smoothl1loss

        fc6_L_kp = mx.sym.CaffeOp(data_0=roi_pool_conv5, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 512}}")
        fc6_U_kp = mx.sym.CaffeOp(data_0=fc6_L_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
        relu6_kp = mx.sym.Activation(data=fc6_U_kp, act_type='relu', name='relu6_kp')
        fc7_L_kp = mx.sym.CaffeOp(data_0=relu6_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 128}}")
        fc7_U_kp = mx.sym.CaffeOp(data_0=fc7_L_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
        relu7_kp = mx.sym.Activation(data=fc6_U_kp, act_type='relu', name='relu7_kp')
        pred_3d = mx.sym.CaffeOp(data_0=relu7_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 64}}")
        #loss_3d = smoothl1loss
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        # weight xavier
        num_stage = 3
        char_stage = 5
        stage_num = ['3', '4', '5']
        stage_char = ['a', 'b', 'c', 'd', 'e']
        for i in range (num_stage - 1):
            for j in range (char_stage - 1):
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv5_1'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu5_1'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv5_2'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu5_2'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv5_3'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv3_1'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu3_1'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv3_2'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu3_2'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv1'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu1'
                arg_params[name + '_weight'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_weight']) * 0.1
                arg_params[name + '_bias'] = mx.nd.ones(shape=self.arg_shape_dict[name + '_bias']) * 0.1
    
    def init_weight_rpn(self, cfg, arg_params, aux_params):
        #arg_params['rpn_conv1_weight'] = 
        arg_params['rpn_conv1_bias'] = mx.nd.ones(shape=self.arg_shape_dict['rpn_conv1_bias']) * 0.1

        #arg_params['rpn_conv1_weight'] = 
        arg_params['rpn_conv1_bias'] = mx.nd.ones(shape=self.arg_shape_dict['rpn_conv1_bias']) * 0.1

        #arg_params['rpn_bbox_pred_fabu_weight'] = 
        arg_params['rpn_bbox_pred_fabu_bias'] = mx.nd.ones(shape=self.arg_shape_dict['rpn_bbox_pred_fabu_bias']) * 0.1
    
    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rcnn(cfg, arg_params, aux_params)
