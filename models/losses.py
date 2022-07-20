import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



###############################################################################
#                                                                             #
#                              Classifiers Losses                             #
#                                                                             #
###############################################################################



class BalancedSoftmaxLoss(nn.CrossEntropyLoss):
    def __init__(self, all_train_label, num_classes, smoothing=0, sample_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert smoothing == 0, str(smoothing)

        if isinstance(all_train_label, torch.Tensor):
            all_train_label = all_train_label.cpu().numpy()

        freq = []
        for i in range(num_classes):
            ids = all_train_label==i
            num = np.sum(ids)
            if num>0:
                if sample_weight is None:
                    freq.append(num)
                else:
                    freq.append(np.sum(sample_weight[ids]))
            else:
                freq.append(0)

        self.register_buffer(
            "class_freq", (torch.Tensor(freq)+1e-6).log().unsqueeze(0))
        # (1, class)

    def forward(self, logit, target):
        logit = logit + self.class_freq
        return super().forward(logit, target)


class BalancedSoftmaxLoss_v2(nn.Module):
    def __init__(self, all_train_label, num_classes, smoothing=0, sample_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_class = num_classes
        self.smoothing = smoothing

        if isinstance(all_train_label, torch.Tensor):
            all_train_label = all_train_label.cpu().numpy()

        freq = []
        for i in range(num_classes):
            ids = all_train_label==i
            num = np.sum(ids)
            if num>0:
                if sample_weight is None:
                    freq.append(num)
                else:
                    freq.append(np.sum(sample_weight[ids]))
            else:
                freq.append(0)

        self.register_buffer(
            "class_freq", (torch.Tensor(freq)+1e-6).log().unsqueeze(0))
        # (1, class)

    def forward(self, logit, target):
        logit = logit + self.class_freq
        with torch.no_grad():
            target_vec = torch.zeros_like(logit)
            target_vec.fill_(self.smoothing / (self.num_class - 1))
            target_vec.scatter_(1, target.unsqueeze(1), 1.0-self.smoothing)
        llh = F.log_softmax(logit, -1)
        loss = torch.mean(torch.sum(-target_vec * llh, dim=-1))
        return loss


class CrossEntropyLossWrapper(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        


def get_classification_loss(name):
    if name == "CE":
        return CrossEntropyLossWrapper
    elif name == "BSM":
        return BalancedSoftmaxLoss
    elif name == "BSMv2":
        return BalancedSoftmaxLoss_v2
    else:
        raise NotImplementedError(name)


###############################################################################
#                                                                             #
#                   Contrast Losses (purity and contrast)                     #
#                                                                             #
###############################################################################

class GmmContrastLoss(nn.Module):
    def __init__(self, contrast_margin=None, reduce="mean"):
        super().__init__()
        self.contrast_margin = contrast_margin
        self.reduce = reduce
        assert reduce in ["mean", "sum", None]

    def compute_sample_loss(self, output_logit, target_logit): 
        raise NotImplementedError("abstract method")
    
    def forward(self, output_logit, target_logit):
        loss = self.compute_sample_loss(output_logit, target_logit)

        if self.reduce == "mean":
            return loss.mean()
        elif self.reduce == "sum":
            return loss.sum()
        elif self.reduce is None:
            return loss



class CrossEntropyContrastLoss(GmmContrastLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_ent = nn.CrossEntropyLoss(reduction='none')

    def compute_sample_loss(self, output_logit, target_logit):
        target_label = target_logit.argmax(-1)
        loss = - self.cross_ent(output_logit, target_label)
        return loss


class SoftCrossEntropyContrastLoss(GmmContrastLoss):
    def compute_sample_loss(self, output_logit, target_logit):
        target_prob = F.softmax(target_logit, -1)
        if self.contrast_margin is not None:
            output_prob = F.softmax(output_logit)
            output_prob = torch.clamp(output_prob, self.contrast_margin, 1)
            neg_ce = target_prob*output_prob.log()
        else:
            neg_ce = target_prob*F.log_softmax(output_logit, -1)

        loss = neg_ce.sum(-1)
        return loss



def get_contrast_loss(name):
    if name == "ce":
        return CrossEntropyContrastLoss
    elif name == "soft_ce":
        return SoftCrossEntropyContrastLoss
    else:
        raise NotImplementedError(name)
