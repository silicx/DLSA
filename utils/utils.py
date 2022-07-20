import math
from collections import OrderedDict
from typing import Any
import numpy as np
import torch
from easydict import EasyDict
import logging


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
       lr = lr / args.warmup_epochs * (epoch + 1 )
    elif args.cos_lr:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1 ) / (args.cls_epoch - args.warmup_epochs + 1 )))
    else:  # stepwise lr schedule
        for milestone in args.steps:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        


def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    logging.info('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)   
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}
    model.load_state_dict(weights)   
    return model


def numpy2torch(item, to_long=False) -> Any:
    """dict of numpy -> torch"""
    if isinstance(item, torch.Tensor):
        if to_long:
            return item.long()
        else:
            return item

    elif isinstance(item, np.ndarray):
        if to_long:
            return torch.from_numpy(item).long()
        else:
            return torch.from_numpy(item)

    elif isinstance(item, list):
        return [numpy2torch(x, to_long) for x in item]

    elif type(item) == dict:
        return {k: numpy2torch(v, to_long) for k,v in item.items()}

    elif type(item) == EasyDict:
        return EasyDict({k: numpy2torch(v, to_long) for k,v in item.items()})

    else:
        raise NotImplementedError(str(type(item)))



def state_dict_strip_prefix(state_dict, prefix="module."):
    state_dict = OrderedDict({
        (k[len(prefix):] if k.startswith(prefix) else k) : v 
        for k,v in state_dict.items()
    })
    return state_dict
