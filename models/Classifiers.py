from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

from models.MLP import MLP
from utils.utils import state_dict_strip_prefix


__all__ = [
    "get_classifier"
]


class CosNorm_Classifier(nn.Module):
    def __init__(self, feat_dim, num_class, scale=16, margin=0.5, *args, **kwargs):
        super(CosNorm_Classifier, self).__init__()

        self.in_dims = feat_dim
        self.out_dims = num_class

        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(self.out_dims, self.in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), dim=1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, dim=1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())




class ClusterAidedClassifier(nn.Module):
    def __init__(self, model_type, feat_dim, num_class):
        super().__init__()

        assert model_type in ["XZP", "XZ"]

        self.model_type = model_type
        self.out_dim = num_class
        self.zp_dim = {
            "XZ": feat_dim,
            "XZP": feat_dim+num_class,
        }[model_type]

        def foo(x):
            raise NotImplementedError()
        self.cls_x = foo
        self.cls_zp = foo


    def forward(self, x, z, p=None):
        if p is None:
            zp = z
        else:
            zp = torch.cat([z, p], 1)
            
        logit_x = self.cls_x(x)
        logit_zp= self.cls_zp(zp)
        logit = logit_x + logit_zp
        return logit




class LinearClusterAidedClassifier(ClusterAidedClassifier):
    def __init__(self, model_type, feat_dim, num_class, weight_path, *args, **kwargs):
        super().__init__(model_type, feat_dim, num_class)

        self.cls_x = MLP(feat_dim, num_class, *args, **kwargs)
        self.cls_zp = MLP(self.zp_dim, num_class, *args, **kwargs)

        if weight_path is not None:
            ckpt = torch.load(weight_path)["state_dict"]
            if "classifier" in ckpt:
                ckpt = ckpt["classifier"]
            ckpt = state_dict_strip_prefix(ckpt)
            print(self.cls_x)
            if "linear.weight" in ckpt:
                self.cls_x.load_state_dict(OrderedDict({
                    "mod.0.weight": ckpt["linear.weight"],
                    "mod.0.bias": ckpt["linear.bias"],
                }))
            elif "linear.weight" in ckpt:
                self.cls_x.load_state_dict(OrderedDict({
                    "mod.0.weight": ckpt["fc.weight"],
                    "mod.0.bias": ckpt["fc.bias"],
                }))
            print(f"loaded {weight_path}")



class CosineClusterAidedClassifier(ClusterAidedClassifier):
    def __init__(self, model_type, feat_dim, num_class, weight_path, *args, **kwargs):
        super().__init__(model_type, feat_dim, num_class)

        self.cls_x = CosNorm_Classifier(feat_dim, num_class, *args, **kwargs)
        self.cls_zp = CosNorm_Classifier(self.zp_dim, num_class, *args, **kwargs)
        
        if weight_path is not None:
            ckpt = torch.load(weight_path)["state_dict"]
            if "classifier" in ckpt:
                ckpt = ckpt["classifier"]
            ckpt = state_dict_strip_prefix(ckpt)
            print(self.cls_x)
            if "linear.weight" in ckpt:
                self.cls_x.load_state_dict(OrderedDict({
                    "weight": ckpt["linear.weight"],
                }))
            elif "linear.weight" in ckpt:
                self.cls_x.load_state_dict(OrderedDict({
                    "weight": ckpt["fc.weight"],
                }))
            print(f"loaded {weight_path}")


def get_classifier(cls_args, args):

    if cls_args.type == "linear":
        ModelClass = LinearClusterAidedClassifier
    elif cls_args.type == "cosine":
        ModelClass = CosineClusterAidedClassifier
    else:
        raise NotImplementedError(cls_args.type)

    return ModelClass(
        cls_args.inputs, args.feat_dim, args.num_class, 
        weight_path=args.backbone_weight_path,
        hidden_layers=cls_args.hidden_layers, batchnorm=cls_args.batch_norm).to(args.device)
