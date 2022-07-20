import torch
import torch.utils.data as tdata
import torch.nn.utils

import copy
import numpy as np
import logging

from utils.metrics import mcc_mni_metrics, shot_acc
from utils.utils import adjust_learning_rate
from models.losses import get_classification_loss



def train_cluster_aided_classifier(
    cls_model, train_loader, test_loader,
    train_label, all_train_label, args):

    cls_args = args.cacls
    
    optimizer = torch.optim.SGD(cls_model.parameters(), 
                                momentum=cls_args.optim.momentum,
                                lr=cls_args.optim.lr)
    
    criterion = get_classification_loss(cls_args.loss.type)(
        train_label, cls_model.out_dim, cls_args.loss.label_smoothing).to(args.device)

    
    best_acc = 0.
    best_epoch = None
    best_state_dict = None
    best_message = None

        
    for epoch in range(1, cls_args.optim.epoch+1):
        cls_model.train()

        adjust_learning_rate(optimizer, epoch, cls_args.optim)

        total_loss = 0
        for X, Z, P, Y in train_loader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            Z = Z.to(args.device)
            P = P.to(args.device)
            logit = cls_model(X, Z, P)
            loss = criterion(logit, Y)
            optimizer.zero_grad()
            loss.backward()
    #         nn.utils.clip_grad_norm_(cls_model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            total_loss += loss.detach().cpu().item()

        logging.info(f"[{epoch}] {total_loss}")
        
        if epoch % cls_args.eval_freq == 0:
            pred, _, gt = eval_cluster_aided_classifier(cls_model, args, test_loader)

            assert pred.shape == gt.shape, str(pred.shape)+", "+str(gt.shape)
            many_shot, median_shot, low_shot = shot_acc(pred, gt, all_train_label)
            mean_acc = np.mean(pred==gt)

            msg = f"Test [{epoch}] {mean_acc*100:.3f} / {many_shot*100:.3f}, {median_shot*100:.3f}, {low_shot*100:.3f}"
            logging.info(msg)

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(cls_model.state_dict())
                best_message = msg

    logging.getLogger("file").info(f"[cacls] Best epoch: {best_epoch}, {best_message}")
    cls_model.load_state_dict(best_state_dict)


    return cls_model



@torch.no_grad()
def eval_cluster_aided_classifier(cls_model, args, test_loader):
    assert isinstance(test_loader, tdata.DataLoader)

    cls_model.eval()
    pred = []
    logit = []
    gt = []
    for X, Z, P, Y in test_loader:
        X, P = X.to(args.device), P.to(args.device)
        Z = Z.to(args.device)
        t = cls_model(X, Z, P)
        logit.append(t)
        pred.append(torch.argmax(t, 1))
        gt.append(Y)
    pred = torch.cat(pred).cpu().numpy()
    logit = torch.cat(logit).cpu().numpy()
    gt = torch.cat(gt).cpu().numpy()
    
    return pred, logit, gt