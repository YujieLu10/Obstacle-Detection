# ---------------------------------------------------------------
# SNIPER: Efficient Multi-scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified from https://github.com/msracver/Deformable-ConvNets
# Modified by Mahyar Najibi
# ---------------------------------------------------------------
import mxnet as mx
from train_utils.lr_scheduler import WarmupMultiBatchScheduler
import os
import logging
import time

def get_optim_params(cfg,roidb_len,batch_size):
    # Create scheduler
    base_lr = cfg.TRAIN.lr
    lr_step = cfg.TRAIN.lr_step
    lr_factor = cfg.TRAIN.lr_factor
    begin_epoch = cfg.TRAIN.begin_epoch
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr_iters = [int(epoch * roidb_len / batch_size) for epoch in lr_epoch_diff]
    if cfg.TRAIN.fp16:
        cfg.TRAIN.warmup_lr /= cfg.TRAIN.scale
    lr_scheduler = WarmupMultiBatchScheduler(lr_iters, lr_factor, cfg.TRAIN.warmup, cfg.TRAIN.warmup_lr, cfg.TRAIN.warmup_step)

    if cfg.TRAIN.fp16 == True:
        optim_params = {'momentum': cfg.TRAIN.momentum,
                        'wd': cfg.TRAIN.wd*cfg.TRAIN.scale,
                        'learning_rate': base_lr/cfg.TRAIN.scale,
                        'rescale_grad': 1.0,
                        'multi_precision': True,
                        'clip_gradient': None,
                        'lr_scheduler': lr_scheduler}
    else:
        optim_params = {'momentum': cfg.TRAIN.momentum,
                        'wd': cfg.TRAIN.wd,
                        'learning_rate': base_lr,
                        'rescale_grad': 1.0,
                        'clip_gradient': None,
                        'lr_scheduler': lr_scheduler}

    return optim_params


def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    paramsmodel = '%s-%04d.params' % (prefix, epoch)
    print '>>>>> paramsmodel'
    print paramsmodel
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))

    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def convert_context(params, ctx):
    """
    :param params: dict of str to NDArray
    :param ctx: the context to convert to
    :return: dict of str of NDArray with context ctx
    """
    new_params = dict()
    for k, v in params.items():
        new_params[k] = v.as_in_context(ctx)
    return new_params


def load_param(prefix, epoch, convert=False, ctx=None, process=False):
    """
    wrapper for load checkpoint
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :param convert: reference model should be converted to GPU NDArray first
    :param ctx: if convert then ctx must be designated.
    :param process: model should drop any test
    :return: (arg_params, aux_params)
    """
    arg_params, aux_params = load_checkpoint(prefix, epoch)
    if convert:
        if ctx is None:
            ctx = mx.cpu()
        arg_params = convert_context(arg_params, ctx)
        aux_params = convert_context(aux_params, ctx)
    if process:
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
    return arg_params, aux_params


def get_fixed_param_names(fixed_param_prefix,sym):
    """
    :param fixed_param_prefix: the prefix in param names to be fixed in the model
    :param fixed_param_prefix: network symbol
    :return: [fixed_param_names]
    """
    fixed_param_names = []
    if fixed_param_prefix is None:
        print('>>>>>> fixed_parma_prefix is none')
        return fixed_param_names
   
    for prefix in fixed_param_prefix:
        print('>>>> prefix')
        print(prefix)
    for name in sym.list_arguments():
        for prefix in fixed_param_prefix:
            if prefix in name:
                fixed_param_names.append(name)
    return fixed_param_names


def create_logger(root_output_path, cfg, image_set):
    # set up logger
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    cfg_name = os.path.basename(cfg).split('.')[0]
    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    if not os.path.exists(config_output_path):
        os.makedirs(config_output_path)

    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    return logger, final_output_path
