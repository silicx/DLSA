import numpy as np
import torch
import torch.utils.data as tdata
from collections import defaultdict
import h5py
import os.path as osp
from easydict import EasyDict

from utils.ClassAwareSampler import ClassAwareSampler
from utils.io import ConfigManager
from utils.utils import numpy2torch



def read_h5_data(args):
    # Read data
    split_name = {"train": "train"}
    if 'iNaturalist' in args.dataset or 'ImageNet' == args.dataset:
        split_name["test"] = 'val'
    else:
        split_name["test"] = 'test'

    features = EasyDict()
    label = EasyDict()

    for split in ["train", "test"]:
        with h5py.File(osp.join(args.feat_dir, f"feature_{split_name[split]}.h5"), "r") as fp:
            for key in fp.keys():
                print(key, fp.get(key).shape)
                
            
            features[split] = torch.from_numpy(fp.get("features")[...]).float().to(args.device)

            if "label" in fp: # compatible
                label[split] = torch.from_numpy(fp.get("label")[...]).float().to(args.device)
            else:
                label[split] = torch.from_numpy(fp.get("labels")[...]).float().to(args.device)


    return features, label




class ContrastiveDataset(tdata.Dataset):
    """Simple dataset"""
    def __init__(self, inputs, labels, weight_q):
        super().__init__()
        print(f"Using simplified sampler with q = {weight_q}")
        self.inputs = inputs.cpu()
        self.labels = labels.cpu()

        self.ids_per_class = defaultdict(list)
        for i,x in enumerate(labels.cpu().numpy().tolist()):
            self.ids_per_class[x].append(i)
        self.ids_per_class = {i:v for i,v in enumerate(self.ids_per_class.values())}

        self.num_classes = len(self.ids_per_class)
        self.num_per_class = [len(self.ids_per_class[i]) for i in range(self.num_classes)]
        max_class_size = np.max(self.num_per_class)*1.0

        self.reverse_freq = np.array([(max_class_size/x)**(weight_q) for x in self.num_per_class])
        # self.reverse_freq = np.minimum(self.reverse_freq, 64)
        self.reverse_freq = self.reverse_freq / np.sum(self.reverse_freq)

        # print(self.reverse_freq)
        print("average weight:", 
            np.sum(self.reverse_freq*np.array(self.num_per_class)) / np.sum(self.num_per_class))

        
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        pos_y, neg_y = np.random.choice(np.arange(self.num_classes), size=2, replace=False).tolist()

        pos_idx  = np.random.choice(self.ids_per_class[pos_y])
        neg_idx  = np.random.choice(self.ids_per_class[neg_y])
        pos_x  = self.inputs[pos_idx]
        neg_x  = self.inputs[neg_idx]

        return (pos_x, neg_x, self.reverse_freq[pos_y])


def make_tensor_dataloader(np_array_list, mask, key, batch_size, shuffle, sampler=None, num_workers=0):
    if mask is not None:
        mask = mask[key]
        tensors = [dt[key][mask, ...] for dt in np_array_list]
    else:
        tensors = [dt[key] for dt in np_array_list]

    tensors = numpy2torch(tensors)

    return tdata.DataLoader(tdata.TensorDataset(*tensors), batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)


def get_data_sampler(sampler_arg: ConfigManager, train_labels):
    if sampler_arg.name is None or sampler_arg.name=="":
        return None
    elif sampler_arg.name == "CBS":
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.cpu().numpy()
        return ClassAwareSampler(train_labels, sampler_arg.num_samples_cls)