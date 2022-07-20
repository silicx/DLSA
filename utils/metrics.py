import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score as NmiMetric
from sklearn.metrics import matthews_corrcoef as MccMetric
import os, json
import numpy as np


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def weighted_shot_acc (preds, labels, ws, train_data, many_shot_thr=100, low_shot_thr=20):
    
    training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(ws[labels==l].sum())
        class_correct.append(((preds[labels==l] == labels[labels==l]) * ws[labels==l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))          
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)



def mcc_mni_metrics(preds, labels):
    return MccMetric(labels, preds), NmiMetric(labels, preds)





def class_count (data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


class SummaryMeter(object):
    def __init__(self, summary_interval, data_type):
        self.summary_interval = summary_interval
        self.values = None
        self.index = 0
        self.data_type = data_type
        assert data_type in ["scalar", "hist"]
    
    def add(self, d, writer):
        if self.values is None:
            self.values = {k:[v] for k,v in d.items()}
            size = 1
        else:
            size = None
            for k,v in d.items():
                self.values[k].append(v)
                size = len(self.values[k])
        
        if size >= self.summary_interval:
            for k,v in self.values.items():
                if self.data_type == "scalar":
                    writer.add_scalar(
                        k, np.mean(v), self.index)
                elif self.data_type == "hist":
                    writer.add_histogram(
                        k, v[-1], self.index)

            self.values = None
            self.index += 1
