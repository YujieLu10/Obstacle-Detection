import mxnet as mx
from symbols.symbol import Symbol
from operator_py.box_annotator_ohem import *
import numpy as np

def checkpoint_callback(bbox_param_names, prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        '''
        print('>>>>> bbox_param_names')
        print(bbox_param_names[0])
        print('>>>>> arg')
        print(arg)
        '''
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
        self.use_global_stats = True
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [64, 256, 512, 1024, 2048]
        

    def get_bbox_param_names(self):
        return ['bbox_pred_fabu_weight', 'bbox_pred_fabu_bias']
    #def get_bbox_param_names(self):
    #    return ['bbox_pred_weight', 'bbox_pred_bias']

    def residual_unit(self, data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512, memonger=False,
                      fix_bn=False):
        if fix_bn or self.fix_bn:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if fix_bn or self.fix_bn:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if fix_bn or self.fix_bn:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
        else:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut

    def residual_unit_dilate(self, data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512,
                             memonger=False):
        if self.fix_bn:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if self.fix_bn:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), dilate=(2, 2),
                                   stride=stride, pad=(2, 2),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if self.fix_bn:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
        else:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut

    def residual_unit_deform(self, data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512,
                             memonger=False):
        if self.fix_bn:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if self.fix_bn:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        offset = mx.symbol.Convolution(name=name + '_offset', data=act2,
                                       num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                       dilate=(2, 2), cudnn_off=True)
        conv2 = mx.contrib.symbol.DeformableConvolution(name=name + '_conv2', data=act2,
                                                        offset=offset,
                                                        num_filter=512, pad=(2, 2), kernel=(3, 3),
                                                        num_deformable_group=4,
                                                        stride=(1, 1), dilate=(2, 2), no_bias=True)
        if self.fix_bn:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, use_global_stats=True, name=name + '_bn3')
        else:
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=self.momentum, name=name + '_bn3')

        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score_fabu = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score_fabu")
        rpn_bbox_pred_fabu = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred_fabu")
        return rpn_cls_score_fabu, rpn_bbox_pred_fabu

    def inc3_unit_left(self, data, name, workspace=512):
        #pad = 0
        conv5_1 = mx.symbol.Convolution(data=data,num_filter=16,kernel=(1,1),stride=(1,1),pad=(0,0),no_bias=True,workspace=workspace,name=name+'/conv5_1')
        relu5_1 = mx.symbol.Activation(data=conv5_1, act_type='relu', name=name+'/relu5_1')
        if 'inc4' in name or 'inc5' in name:
            conv5_2 = mx.symbol.Convolution(data=relu5_1, num_filter=32,kernel=(3,3),stride=(1,1),dilate=(2,2),pad=(2,2),no_bias=True,workspace=workspace,name=name+'/conv5_2')
            relu5_2 = mx.symbol.Activation(data=conv5_2, act_type='relu', name=name+'/relu5_2')
            conv5_3 = mx.symbol.Convolution(data=relu5_2, num_filter=32,kernel=(3,3),stride=(1,1),dilate=(2,2),pad=(2,2),no_bias=True,workspace=workspace,name=name+'/conv5_3')
            relu5_3 = mx.symbol.Activation(data=conv5_3, act_type='relu', name=name+'/relu5_3')
        else:
            conv5_2 = mx.symbol.Convolution(data=relu5_1, num_filter=32,kernel=(3,3),stride=(1,1),pad=(1,1),no_bias=True,workspace=workspace,name=name+'/conv5_2')
            relu5_2 = mx.symbol.Activation(data=conv5_2, act_type='relu', name=name+'/relu5_2')
            if 'inc3a' in name:
                conv5_3 = mx.symbol.Convolution(data=relu5_2, num_filter=32,kernel=(3,3),stride=(2,2),pad=(1,1),no_bias=True,workspace=workspace,name=name+'/conv5_3')
            else:
                conv5_3 = mx.symbol.Convolution(data=relu5_2, num_filter=32,kernel=(3,3),stride=(1,1),pad=(1,1),no_bias=True,workspace=workspace,name=name+'/conv5_3')
            relu5_3 = mx.symbol.Activation(data=conv5_3, act_type='relu', name=name+'/relu5_3')
        return relu5_3

    def inc3_unit_middle(self, data, name, workspace=512):
        #pad = 0
        conv3_1 = mx.symbol.Convolution(data=data, num_filter=16,kernel=(1,1),stride=(1,1),pad=(0,0),no_bias=True,workspace=workspace,name=name+'/conv3_1')
        relu3_1 = mx.symbol.Activation(data=conv3_1,act_type='relu', name=name+'/relu3_1')
        '''
        if name == 'inc3a':
            incstride = (1,1)
        else:
            incstride = (1,1)
        '''
        incstride = (1,1)
        #dilate=(2,2)
        if 'inc4' in name or 'inc5' in name:
            conv3_2 = mx.symbol.Convolution(data=relu3_1, num_filter=64,kernel=(3,3),stride=incstride,dilate=(2,2),pad=(2,2),no_bias=True,workspace=workspace,name=name+'/conv3_2')
            relu3_2 = mx.symbol.Activation(data=conv3_2,act_type='relu', name=name+'/relu3_2')
        else:        
            conv3_2 = mx.symbol.Convolution(data=relu3_1, num_filter=64,kernel=(3,3),stride=incstride,pad=(1,1),no_bias=True,workspace=workspace,name=name+'/conv3_2')
            relu3_2 = mx.symbol.Activation(data=conv3_2,act_type='relu', name=name+'/relu3_2')
        return relu3_2

    def inc3_unit_right(self, data, num_output, name, workspace=512):
        #pad = (0,0)
        #kernel=(1,1)
        #print('>>>>>>> inc3_unit_right')
        #print(name)
        conv1 = mx.symbol.Convolution(data=data, num_filter=num_output,kernel=(1,1),stride=(1,1),pad=(0,0),no_bias=True,workspace=workspace,name=name+'/conv1')
        relu1 = mx.symbol.Activation(data=conv1, act_type='relu', name=name+'/relu1')
        return relu1

    def get_symbol_rcnn(self, cfg, is_train=True):
        num_anchors = cfg.network.NUM_ANCHORS
        #yujie
        workspace = self.workspace
        num_stage = 3
        char_stage = 5
        stage_num = ['3', '4', '5']
        stage_char = ['a', 'b', 'c', 'd', 'e']
        #filter_list = [96, 128, 128]
        filter_list = [96, 128, 128]
        #
        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name='gt_boxes')
            valid_ranges = mx.sym.Variable(name='valid_ranges')
            im_info = mx.sym.Variable(name='im_info')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name='im_info')
            im_ids = mx.sym.Variable(name='im_ids')
        # shared convolutional layers
        
        #yujie
        '''
        conv1 = mx.symbol.Convolution(data=data, num_filter = 32, kernel=(4, 4), stride=(2, 2), pad=(1, 1), no_bias=True, workspace=workspace, name='conv1')

        relu1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu1')
        
        conv2 = mx.symbol.Convolution(data=relu1, num_filter = 48, kernel=(3, 3),
        stride=(2, 2), pad=(1, 1), no_bias=True, workspace=workspace, name='conv2')

        relu2 = mx.symbol.Activation(data=conv2, act_type='relu', name='relu2')

        conv3 = mx.symbol.Convolution(data=relu2, num_filter = 96, kernel=(3, 3),
        stride=(2, 2), pad=(1, 1), no_bias=True, workspace=workspace, name='conv3')

        relu3 = mx.symbol.Activation(data=conv3, act_type='relu', name='relu3')

        #stride=(1, 1) yujie
        conv3_down2 = mx.symbol.Pooling(data=conv3, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
        inc3e = conv3
        #incleft
        for i in range (num_stage):
            print(stage_num[i])
            #inc3/pool
            #stride=(2 if i == 0 else 1, 2 if i == 0 else 1)
            #inc3_right = mx.symbol.Pooling(data=conv3 if i==0 else inc_concat, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max')
            #inc3_right = mx.symbol.Pooling(data=conv3 if i == 0 else inc_concat, kernel=(2 if i == 0 else 3, 2 if i == 0 else 3), stride=(2,2), pad=(0 if i==0 else 1, 0 if i==0 else 1), pool_type='max')
            for j in range (char_stage):
                print(stage_char[j])
                inc3_right = mx.symbol.Pooling(data=conv3 if i==0 else inc_concat, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max')
                inc3_left = self.inc3_unit_left(conv3 if i == 0 else inc_concat, 'inc' + stage_num[i] + stage_char[j], workspace)
                inc3_middle = self.inc3_unit_middle(relu3 if i == 0 else inc_concat, 'inc' + stage_num[i] + stage_char[j], workspace)
                #inc3_right if j == 0 else inc_concat
                inc3_right = self.inc3_unit_right(inc3_right, filter_list[i], 'inc' + stage_num[i] + stage_char[j], workspace)
                #print('>>>>>>  inc3_left')
                #print(inc3_left.shape, inc3_middle.shape, inc3_right.shape)
                inc_concat = mx.symbol.concat(inc3_left, inc3_middle, inc3_right)
                #inc_concat = mx.symbol.concat(*[inc3_left, inc3_middle, inc3_right])
                #inc_concat = mx.symbol.concat(*[inc3_left, inc3_middle])
                #inc_concat = mx.symbol.concat(*[inc3_left, inc3_right])
                #inc_concat = mx.symbol.concat(*[inc3_middle, inc3_right])
                #print('>>>>> inc_concat')
                #mx.visualization.print_summary(inc_concat)
                #mx.visualization.print_summary(inc_concat,{"data":(1,3,1056,640),"gt_boxes": (1, 100, 5), "label": (1, 23760), "bbox_target": (1, 36, 66, 40), "bbox_weight": (1, 36, 66, 40)})
                #inc_concat = mx.symbol.concat()
                if stage_char[j] == 'e' and stage_num[i] == '3':
                    inc3e = inc_concat
                    print('>>>>> inc3e = inc_concat')
        print('>>>>> rcnn conv3_down2') 
        #mx.visualization.print_summary(conv3_down2)
        #concat
        inc_concat = mx.symbol.Pooling(data=inc_concat, kernel=(3,3), stride=(2,2), pad=(1,1), pool_type='max')
        inc3e = mx.symbol.Pooling(data=inc3e, kernel=(3,3), stride=(2,2), pad=(1,1), pool_type='max')
        print('>>>>> rcnn concat conv3_down2 inc_concat inc3e')
        concat = mx.symbol.concat(*[conv3_down2, inc_concat, inc3e])
        #concat = mx.symbol.concat(*[conv3_down2, inc_concat])
        #concat = inc_concat
        #mx.visualization.print_summary(concat)
        #convf/reluf
        #convf = mx.symbol.Convolution(data = concat, num_filter=256, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias = True,  workspace=self.workspace, name='convf')
        #reluf = mx.symbol.Activation(data = convf, act_type='relu', name='reluf')
        #
        '''
        
        conv_feat = self.resnetc4(data, fp16=cfg.TRAIN.fp16)
        # res5
        relut = self.resnetc5(conv_feat, deform=True)
        relu1 = mx.symbol.Concat(*[conv_feat, relut], name='cat4')
        if cfg.TRAIN.fp16:
            relu1 = mx.sym.Cast(data=relu1, dtype=np.float32)        
        #yujie

        #num_filter = 98
        '''
        rpn_cls_score_fabu = mx.symbol.Convolution(data = relu1, kernel=(1,1),pad=(0,0),num_filter=2 * num_anchors,name='rpn_cls_score_fabu')
        #num_filter=196
        #stride 2
        rpn_bbox_pred_fabu = mx.symbol.Convolution(data=relu1,kernel=(1,1),pad=(0,0),num_filter=4 * num_anchors,name='rpn_bbox_pred_fabu')
        '''
        rpn_cls_score_fabu, rpn_bbox_pred_fabu = self.get_rpn(relu1, num_anchors)

        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')
        rpn_cls_score_fabu_reshape = mx.sym.Reshape(data=rpn_cls_score_fabu, shape = (0,2, -1, 0), name = "rpn_cls_score_fabu_reshape")
        # generate anchor ?
        #rpn_data = mx.nd.contrib.MultiBoxPrior(data = data, sizes=[1.5, 3, 6, 9, 16, 32, 48],ratios=[0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0],steps=[16,16],name='rpn_data')
        #rpn_bbox_loss
        #data=(rpn_bbox_pred_fabu - rpn_bbox_target)
        #rpn_bbox_loss = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss',scalar=1.0,data=(rpn_bbox_pred_fabu - rpn_bbox_target))
        #rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2, -1, 0),name='rpn_cls_prob_reshape')
    
        if is_train:
            #data=rpn_cls_score_reshape
                    #grad_scale
            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0
            rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_fabu_reshape, label=rpn_label, multi_output=True, normalization='valid',use_ignore=True, ignore_label=-1,name='rpn_cls_prob',grad_scale=grad_scale)
            
            #rpn_cls_prob
            #shae=(0, 98, -1, 0)
            proposal, label, bbox_target, bbox_weight = mx.symbol.MultiProposalTarget(cls_prob=rpn_cls_prob,bbox_pred=rpn_bbox_pred_fabu, im_info=im_info, gt_boxes=gt_boxes, valid_ranges=valid_ranges,  batch_size=cfg.TRAIN.BATCH_IMAGES, name='multi_proposal_target')

            #print('>>>>>>> rpn_bbox_pred_fabu {}'.format(rpn_bbox_pred_fabu))
            #print('>>>>>>> rpn_bbox_target  {}'.format(rpn_bbox_target))
            #print('>>>>>>> proposal  {}'.format(proposal))
            #add from resnet_mx
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')

            #rcnn_label = label
            offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=proposal, group_size=1, pooled_size=7, sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
            #yujie lr_mult = 0.01
            offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.0001)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")
            
            deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=conv_new_1_relu, rois=proposal,trans=offset_reshape, group_size=1, pooled_size=7, sample_per_part=4, no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)

            fc6_L = mx.symbol.FullyConnected(name='fc6_L', data=deformable_roi_pool, num_hidden=512)
            #fc6_L = mx.symbol.CaffeOp(data_0=roi_pool_conv5, prototxt="layer {type:\"InnerProduct\" inner_product_param {num_output: 512}}")
            fc6_U = mx.symbol.FullyConnected(name = 'fc6_U', data = fc6_L, num_hidden=4096)
            #fc6_U = mx.symbol.CaffeOp(data_0=fc6_L, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
            relu6 = mx.symbol.Activation(data = fc6_U, act_type = 'relu', name = 'relu6')

            fc7_L = mx.symbol.FullyConnected(name = 'fc7_L', data = relu6, num_hidden=128)
            #fc7_L = mx.symbol.CaffeOp(data_0=relu6, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 128}}")
            fc7_U = mx.symbol.FullyConnected(name = 'fc7_U', data = fc7_L, num_hidden=4096)
            #fc7_U = mx.symbol.CaffeOp(data_0=fc7_L, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
            #relu7 = mx.symbol.FullyConnected(name = 'relu7', data = fc7_U)
            relu7 = mx.symbol.Activation(data=fc7_U, act_type = 'relu', name='relu7')
            num_classes = 27 
            num_reg_classes = 1
            cls_score_fabu = mx.symbol.FullyConnected(name = 'cls_score_fabu', data = fc7_U, num_hidden=num_classes)
            
            #bbox_pred_fabu = mx.symbol.FullyConnected(name = 'bbox_pred_fabu', data = relu7, num_hidden=32)
            bbox_pred_fabu = mx.symbol.FullyConnected(name = 'bbox_pred_fabu', data = relu7, num_hidden=num_reg_classes * 4)

 
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score_fabu, label=label, normalization='valid', use_ignore=True, ignore_label=-1, grad_scale=grad_scale)

            loss_bbox_ = bbox_weight * mx.sym.smooth_l1(name='loss_bbox_', scalar=1.0, data=(bbox_pred_fabu - bbox_target))
            loss_bbox = mx.sym.MakeLoss(name='loss_bbox', data=loss_bbox_, grad_scale=grad_scale/(188.0*16.0))
            rcnn_label = label

            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            
            loss_bbox = mx.sym.Reshape(data=loss_bbox, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='loss_bbox_reshape')

            rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=1.0, data=(rpn_bbox_pred_fabu - rpn_bbox_target))

            rpn_bbox_loss = mx.symbol.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,grad_scale=3*grad_scale / float(cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))
            group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, loss_bbox, mx.sym.BlockGrad(rcnn_label)])

        else:
        #batchsize?? self.test_nbatch
            rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_fabu_reshape, mode="channel", name='rpn_cls_prob')
            rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            proposal, _ = mx.symbol.MultiProposal(cls_prob=rpn_cls_prob_reshape,bbox_pred=rpn_bbox_pred_fabu, im_info=im_info,name='proposal', 
            batch_size=self.test_nbatch,rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N,rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
            rpn_min_size=cfg.TEST.RPN_MIN_SIZE,threshold=cfg.TEST.RPN_NMS_THRESH,feature_stride=cfg.network.RPN_FEAT_STRIDE,ratios=tuple(cfg.network.ANCHOR_RATIOS),scales=tuple(cfg.network.ANCHOR_SCALES))

            offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=proposal,
                                                             group_size=1, pooled_size=7,
                                                             sample_per_part=4, no_trans=True, part_size=7,
                                                             output_dim=256, spatial_scale=0.0625)
            #yujie lr_mult=0.01                                                             
            offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.0001)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

            deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool',
                                                                        data=conv_new_1_relu, rois=proposal,
                                                                        trans=offset_reshape, group_size=1,
                                                                        pooled_size=7, sample_per_part=4,
                                                                        no_trans=False, part_size=7, output_dim=256,
                                                                        spatial_scale=0.0625, trans_std=0.1)

            fc6_L = mx.symbol.FullyConnected(name='fc6_L', data=deformable_roi_pool, num_hidden=512)
            #fc6_L = mx.symbol.CaffeOp(data_0=roi_pool_conv5, prototxt="layer {type:\"InnerProduct\" inner_product_param {num_output: 512}}")
            fc6_U = mx.symbol.FullyConnected(name = 'fc6_U', data = fc6_L, num_hidden=4096)
            #fc6_U = mx.symbol.CaffeOp(data_0=fc6_L, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
            relu6 = mx.symbol.Activation(data = fc6_U, act_type = 'relu', name = 'relu6')

            fc7_L = mx.symbol.FullyConnected(name = 'fc7_L', data = relu6, num_hidden=128)
            #fc7_L = mx.symbol.CaffeOp(data_0=relu6, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 128}}")
            fc7_U = mx.symbol.FullyConnected(name = 'fc7_U', data = fc7_L, num_hidden=4096)
            #fc7_U = mx.symbol.CaffeOp(data_0=fc7_L, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
            #relu7 = mx.symbol.FullyConnected(name = 'relu7', data = fc7_U)
            relu7 = mx.symbol.Activation(data=fc7_U, act_type = 'relu', name='relu7')
            num_classes = 27
            num_reg_classes = 1
            cls_score_fabu = mx.symbol.FullyConnected(name = 'cls_score_fabu', data = fc7_U, num_hidden=num_classes)
            
            #bbox_pred_fabu = mx.symbol.FullyConnected(name = 'bbox_pred_fabu', data = relu7, num_hidden=32)
            bbox_pred_fabu = mx.symbol.FullyConnected(name = 'bbox_pred_fabu', data = relu7, num_hidden=num_reg_classes * 4)
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score_fabu)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(self.test_nbatch, -1, num_classes), name='cls_prob_reshape')
            bbox_pred_fabu = mx.sym.Reshape(data=bbox_pred_fabu, shape=(self.test_nbatch, -1, 4 * num_reg_classes), name='bbox_pred_fabu_reshape')
            group = mx.symbol.Group([proposal, cls_prob, bbox_pred_fabu, im_ids])
            '''
            batch_size=16,
            rpn_pre_nms_top_n=10000,rpn_post_nms_top_n=2000, rpn_min_size=8, threshold=0.7,feature_stride=16,ratios=(0.5, 1, 2), scales=(2, 4, 7, 10, 13, 16, 24))
            '''            
        #rpn_loss_cls =
        #mute_rpn_scores = ?

        #roi_data python?
        #roi_data = 
        #roi_pool_conv5 = mx.symbol.ROIPooling(data = reluf, rois = proposal, pooled_size = (6, 6), spatial_scale=0.0625, name = 'roi_pool_conv5')


            
        #cls_score_fabu = mx.symbol.CaffeOp(data_0=fc7_U, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 8}}")
        #bbox_pred_fabu = mx.symbol.CaffeOp(data_0=relu7, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 32}}")
        
        #loss_cls = softmaxwithloss
        #label -> labels -> roi-data/labels

        #loss_cls = 
        #loss_cls = mx.symbol.CaffeLoss(data = cls_score_fabu, label = label, grad_scale = 1, name='loss_cls', prototxt="layer{type:\"SoftmaxWithLoss\"}")
        #num_weight=?
        #loss_bbox = ? smoothl1loss
        '''
        fc6_L_kp = mx.symbol.FullyConnected(name = 'fc6_L_kp', data = fc7_U, num_hidden=512)
        #fc6_L_kp = mx.symbol.CaffeOp(data_0=roi_pool_conv5, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 512}}")
        fc6_U_kp = mx.symbol.FullyConnected(name = 'fc6_U_kp', data = fc6_L_kp, num_hidden=4096)
        #fc6_U_kp = mx.symbol.CaffeOp(data_0=fc6_L_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
        relu6_kp = mx.symbol.Activation(data=fc6_U_kp, act_type='relu', name='relu6_kp')
        fc7_L_kp = mx.symbol.FullyConnected(name = 'fc7_L_kp', data = relu6_kp, num_hidden=128)
        #fc7_L_kp = mx.symbol.CaffeOp(data_0=relu6_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 128}}")
        fc7_U_kp = mx.symbol.FullyConnected(name = 'fc7_U_kp', data = fc7_L_kp, num_hidden=4096)
        #fc7_U_kp = mx.symbol.CaffeOp(data_0=fc7_L_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 4096}}")
        relu7_kp = mx.symbol.Activation(data=fc7_U_kp, act_type='relu', name='relu7_kp')
        pred_3d = mx.symbol.FullyConnected(name = 'pred_3d', data = relu7_kp, num_hidden=64)
        #pred_3d = mx.symbol.CaffeOp(data_0=relu7_kp, prototxt="layer {type: \"InnerProduct\"inner_product_param {num_output: 64}}")
        #loss_3d = smoothl1loss
        '''
            
        self.sym = group
        return group
    '''
    def get_symbol_rpn(self, cfg, is_train=True):
        workspace = self.workspace
        num_stage = 3
        char_stage = 5
        stage_num = ['3', '4', '5']
        stage_char = ['a', 'b', 'c', 'd', 'e']
        #filter_list = [96, 128, 128]
        filter_list = [96, 128, 128]
        if is_train:
            data = mx.symbol.Variable(name="data")
            rpn_label = mx.symbol.Variable(name='label')
            rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
            rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')
            gt_boxes = mx.symbol.Variable(name='gt_boxes')
            valid_ranges = mx.symbol.Variable(name='valid_ranges')
            im_info = mx.symbol.Variable(name='im_info')
        else:
            data = mx.symbol.Variable(name="data")
            im_info = mx.symbol.Variable(name='im_info')
            im_ids = mx.symbol.Variable(name='im_ids')

        conv1 = mx.symbol.Convolution(data=data, num_filter = 32, kernel=(4, 4), stride=(2, 2), pad=(1, 1), no_bias=True, workspace=workspace, name='conv1')

        relu1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu1')
        
        conv2 = mx.symbol.Convolution(data=relu1, num_filter = 48, kernel=(3, 3),
        stride=(2, 2), pad=(1, 1), no_bias=True, workspace=workspace, name='conv2')

        relu2 = mx.symbol.Activation(data=conv2, act_type='relu', name='relu2')

        conv3 = mx.symbol.Convolution(data=relu2, num_filter = 96, kernel=(3, 3),
        stride=(2, 2), pad=(1, 1), no_bias=True, workspace=workspace, name='conv3')

        relu3 = mx.symbol.Activation(data=conv3, act_type='relu', name='relu3')

        #stride=(2, 2)
        conv3_down2 = mx.symbol.Pooling(data=conv3, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max')
        inc3e = conv3
        #incleft
        for i in range (num_stage):
            print(stage_num[i])
            for j in range (char_stage):
                print(stage_char[j])
                inc3_right = mx.symbol.Pooling(data=conv3 if i==0 else inc_concat, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max')
                inc3_left = self.inc3_unit_left(conv3 if i == 0 else inc_concat, 'inc' + stage_num[i] + stage_char[j], workspace)
                inc3_middle = self.inc3_unit_middle(relu3 if i == 0 else inc_concat, 'inc' + stage_num[i] + stage_char[j], workspace)
                #inc3_right if j == 0 else inc_concat
                inc3_right = self.inc3_unit_right(inc3_right, filter_list[i], 'inc' + stage_num[i] + stage_char[j], workspace)
                inc_concat = mx.symbol.concat(inc3_left, inc3_middle, inc3_right)
                if stage_char[j] == 'e' and stage_num[i] == '3':
                    inc3e = inc_concat
                    print('>>>>> inc3e = inc_concat')
        print('>>>>> rcnn conv3_down2') 
        #mx.visualization.print_summary(conv3_down2)
        #concat
        print('>>>>> rcnn concat conv3_down2 inc_concat inc3e')
        concat = mx.symbol.concat(*[conv3_down2, inc_concat, inc3e])
        #concat = mx.symbol.concat(*[conv3_down2, inc_concat])
        #concat = inc_concat
        #mx.visualization.print_summary(concat)
        #convf/reluf
        convf = mx.symbol.Convolution(data = concat, num_filter=256, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias = True,  workspace=self.workspace, name='convf')
        reluf = mx.symbol.Activation(data = convf, act_type='relu', name='reluf')

        #grad_scale
        if cfg.TRAIN.fp16 == True:
            grad_scale = float(cfg.TRAIN.scale)
        else:
            grad_scale = 1.0

        rpn_conv1 = mx.symbol.Convolution(data = convf, kernel=(1,1),pad=(0,0),num_filter=256,name='rpn_conv1')
        rpn_relu1 = mx.symbol.Activation(data = rpn_conv1, act_type='relu', name='rpn_relu1')
        #num_filter = 98
        rpn_cls_score_fabu = mx.symbol.Convolution(data = rpn_relu1, kernel=(1,1),pad=(0,0),num_filter=42,name='rpn_cls_score_fabu')
        if is_train:
            rpn_cls_score_reshape = mx.symbol.Reshape(data = rpn_cls_score_fabu, shape=(0,8,-1,0),name='rpn_cls_score_reshape')
        else:
            rpn_cls_score_reshape = mx.symbol.Reshape(data = rpn_cls_score_fabu, shape=(0,2,-1,0),name='rpn_cls_score_reshape')
        #num_filter=196
        rpn_bbox_pred_fabu = mx.symbol.Convolution(data=rpn_relu1, stride=(2,2),kernel=(1,1),pad=(0,0),num_filter=84,name='rpn_bbox_pred_fabu')

        # generate anchor ?
        #rpn_data = mx.nd.contrib.MultiBoxPrior(data = data, sizes=[1.5, 3, 6, 9, 16, 32, 48],ratios=[0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0],steps=[16,16],name='rpn_data')
        #rpn_bbox_loss
        #data=(rpn_bbox_pred_fabu - rpn_bbox_target)
        #rpn_bbox_loss = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss',scalar=1.0,data=(rpn_bbox_pred_fabu - rpn_bbox_target))
        if is_train:

            #data=rpn_cls_score_reshape
            rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True, normalization='valid',use_ignore=True, ignore_label=-1,name='rpn_cls_prob',grad_scale=grad_scale)
            #rpn_cls_prob
            #shae=(0, 98, -1, 0)
            rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2, -1, 0),name='rpn_cls_prob_reshape')
            
            proposal, rpn_scores = mx.symbol.MultiProposal(cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred_fabu, im_info=im_info, name='proposal', batch_size=16, rpn_pre_nms_top_n=10000, rpn_post_nms_top_n=2000, rpn_min_size=8, threshold=0.7, feature_stride=16, ratios=(0.5, 1, 2), scales=(2, 4, 7, 10, 13, 16, 24))
            
            #rpn_bbox_pred_fabu - rpn_bbox_target
            rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=1.0, data=(rpn_bbox_pred_fabu - rpn_bbox_target))

            rpn_bbox_loss = mx.symbol.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,grad_scale=3*grad_scale / float(cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))
            
            #add from resnet_mx
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            rcnn_label = label


        else:
        #batchsize?? self.test_nbatch
            rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel", name='rpn_cls_prob')
            rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2, -1, 0), name='rpn_cls_prob_reshape')
            proposal, rpn_scores = mx.symbol.MultiProposal(cls_prob=rpn_cls_prob_reshape,bbox_pred=rpn_bbox_pred_fabu, im_info=im_info,name='proposal', batch_size=16, rpn_pre_nms_top_n=10000,rpn_post_nms_top_n=2000, rpn_min_size=8, threshold=0.7,feature_stride=16,ratios=(0.5, 1, 2), scales=(2, 4, 7, 10, 13, 16, 24))

        if is_train:
            group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss])
        else:
            #group = mx.symbol.Group([proposal, rpn_cls_prob, bbox_pred_fabu, im_ids])
            group = mx.symbol.Group([proposal, rpn_scores, im_ids])
        self.sym = group
        return group
    '''

    def resnetc4(self, data, fp16=False):
        units = self.units
        filter_list = self.filter_list
        bn_mom = self.momentum
        workspace = self.workspace
        num_stage = len(units)
        memonger = False

        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, use_global_stats=True, name='bn_data')
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        if fp16:
            body = mx.sym.Cast(data=body, dtype=np.float16)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, use_global_stats=True, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

        for i in range(num_stage - 1):
            body = self.residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                                      name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                                      memonger=memonger, fix_bn=(i == 0))
            for j in range(units[i] - 1):
                body = self.residual_unit(body, filter_list[i + 1], (1, 1), True,
                                          name='stage%d_unit%d' % (i + 1, j + 2),
                                          workspace=workspace, memonger=memonger, fix_bn=(i == 0))

        return body

    def resnetc5(self, body, deform):
        units = self.units
        filter_list = self.filter_list
        workspace = self.workspace
        num_stage = len(units)
        memonger = False

        i = num_stage - 1
        if deform:
            body = self.residual_unit_deform(body, filter_list[i + 1], (1, 1), False,
                                             name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                                             memonger=memonger)
        else:
            body = self.residual_unit_dilate(body, filter_list[i + 1], (1, 1), False,
                                             name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                                             memonger=memonger)
        for j in range(units[i] - 1):
            if deform:
                body = self.residual_unit_deform(body, filter_list[i + 1], (1, 1), True,
                                                 name='stage%d_unit%d' % (i + 1, j + 2),
                                                 workspace=workspace, memonger=memonger)
            else:
                body = self.residual_unit_dilate(body, filter_list[i + 1], (1, 1), True,
                                                 name='stage%d_unit%d' % (i + 1, j + 2),
                                                 workspace=workspace, memonger=memonger)

        return body

    def add_arg(self, arg_params, name):
        if self.arg_shape_dict.has_key(name + '_weight'):
            print('>>>>>> add arg {}'.format(name+'weight'))
            arg_params[name + '_weight'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_weight'])
        if self.arg_shape_dict.has_key(name + '_bias'):
            print('>>>>>> add arg {}'.format(name+'bias'))
            arg_params[name + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_bias'])

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        # weight xavier
        num_stage = 3
        char_stage = 5
        #print('>>>>>>> rcnn arg_parmas {}'.format(arg_params.keys()))
        stage_num = ['3', '4', '5']
        stage_char = ['a', 'b', 'c', 'd', 'e']
        '''
        arg_params['stage4_unit1_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_weight'])
        arg_params['stage4_unit1_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_bias'])
        arg_params['stage4_unit2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_weight'])
        arg_params['stage4_unit2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_bias'])
        arg_params['stage4_unit3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_weight'])
        arg_params['stage4_unit3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_bias'])
        '''
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])

        arg_params['cls_score_fabu_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_fabu_weight'])
        arg_params['cls_score_fabu_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_fabu_bias'])

        arg_params['rpn_cls_score_fabu_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_fabu_weight'])
        arg_params['rpn_cls_score_fabu_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_fabu_bias'])

        arg_params['rpn_bbox_pred_fabu_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_fabu_weight'])
        arg_params['rpn_bbox_pred_fabu_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_fabu_bias'])

        #arg_params['convf_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['convf_weight'])
        #arg_params['convf_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['convf_bias'])
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_weight'])
        arg_params['offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_bias'])

        arg_params['fc6_L_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc6_L_weight'])
        arg_params['fc6_L_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_L_bias'])
        arg_params['fc6_U_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc6_U_weight'])
        arg_params['fc6_U_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_U_bias'])
        arg_params['fc7_L_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc7_L_weight'])
        arg_params['fc7_L_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc7_L_bias'])
        arg_params['fc7_U_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc7_U_weight'])
        arg_params['fc7_U_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc7_U_bias'])
        #print('>>>>> arg_shape_dict')
        #print(self.arg_shape_dict)
        arg_params['bbox_pred_fabu_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_fabu_weight'])
        arg_params['bbox_pred_fabu_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_fabu_bias'])
        #print('>>>>>> self.arg_shape_dict {}'.format(self.arg_shape_dict.keys()))
        
        for i in range (num_stage - 1):
            for j in range (char_stage - 1):
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv5_1'
                #arg_params[name + '_weight'] = arg_params['conv_new_1_weight']
                #arg_params[name + '_bias'] = arg_params['conv_new_1_bias']
                #self.arg_shape_dict[name + '_weight'] = self.arg_shape_dict['conv_new_1_weight']
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu5_1'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv5_2'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu5_2'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv5_3'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv3_1'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu3_1'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv3_2'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu3_2'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/conv1'
                self.add_arg(arg_params, name)
                name = 'inc' + stage_num[i] + stage_char[j] + '/relu1'
                self.add_arg(arg_params, name)
        
        
    '''
    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        
        arg_params['stage4_unit1_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_weight'])
        arg_params['stage4_unit1_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_bias'])
        arg_params['stage4_unit2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_weight'])
        arg_params['stage4_unit2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_bias'])
        arg_params['stage4_unit3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_weight'])
        arg_params['stage4_unit3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_bias'])
        
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_weight'])
        arg_params['offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])
    '''    

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        #print('>>>>>> rpn arg_params {}'.format(arg_params))
        arg_params['stage4_unit1_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_weight'])
        arg_params['stage4_unit1_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_offset_bias'])
        arg_params['stage4_unit2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_weight'])
        arg_params['stage4_unit2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_offset_bias'])
        arg_params['stage4_unit3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_weight'])
        arg_params['stage4_unit3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_offset_bias'])

        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])

        arg_params['rpn_cls_score_fabu_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_fabu_weight'])
        arg_params['rpn_cls_score_fabu_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_fabu_bias'])
        arg_params['rpn_bbox_pred_fabu_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_fabu_weight'])
        arg_params['rpn_bbox_pred_fabu_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_fabu_bias'])
        #arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        #arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
    
    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rcnn(cfg, arg_params, aux_params)
