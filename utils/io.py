import argparse
import glob
import importlib.util
import logging
import os
import shutil
from typing import Any, Optional
import h5py
import os.path as osp
import torch
import yaml
from torch.utils.tensorboard.writer import SummaryWriter




class ConfigManager(object):
    """Experiment configuration container, whose values can be access as property or by indexing.
    * Case insensitive

    # Initialize
    ConfigManager():  initalize an empty config
    ConfigManager.init_with_dict(dict)
    ConfigManager.init_with_yaml(yaml path)
    ConfigManager.init_with_namespace(namespace)

    # Subset of configs
    cfg.subset(*keys)
    cfg.except(*keys)

    # Update
    cfg.override_with_dict(dict)
    cfg.override_with_yaml(yaml path)
    cfg.override_with_namespace(namespace)
    """
    def __init__(self):
        self.__cfg = {}

    @classmethod
    def init_with_dict(cls, dict_cfg: dict):
        assert isinstance(dict_cfg, dict)

        instance = cls()
        for key, value in dict_cfg.items():
            key = key.lower()
            if isinstance(value, dict):
                value = cls.init_with_dict(dict_cfg=value)
            instance.__cfg[key] = value
        return instance

    @classmethod
    def init_with_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as fp:
            a_dict = yaml.load(fp, yaml.FullLoader)
        return cls.init_with_dict(dict_cfg=a_dict)
    
    def subset(self, *keys: str):
        include_keys = {x.lower() for x in keys}
        cfg = ConfigManager()
        cfg.__cfg = {k:v for k,v in self.__cfg.items() if k in include_keys}
        return cfg

    def excepts(self, *keys: str):
        exclude_keys = {x.lower() for x in keys}
        cfg = ConfigManager()
        cfg.__cfg = {k:v for k,v in self.__cfg.items() if k not in exclude_keys}
        return cfg

    def override_with_dict(self, cfg: dict, strict: bool=True):
        """override configs with a `dict`"""
        assert isinstance(cfg, dict)

        for key, value in cfg.items():
            key = key.lower()
            if key not in self.__cfg:
                if strict:
                    raise KeyError(key)
                else:
                    self.__cfg[key] = ConfigManager()

            if isinstance(value, dict):
                self.__cfg[key].override_with_dict(value, strict)
            else:
                self.__cfg[key] = value

        
    def override_with_yaml(self, yaml_path: str, strict: bool=True):
        """override configs with a yaml file"""
        with open(yaml_path, "r") as fp:
            a_dict = yaml.load(fp, yaml.FullLoader)
            self.override_with_dict(a_dict, strict)

    def update(self, key: str, new_val: Any):
        if "." in key:
            key_lst = key.split(".")
            key_this, key_rest = key_lst[0], ".".join(key_lst[1:])
            self.__cfg[key_this].update(key_rest, new_val)
        else:
            self.__cfg[key] = new_val



    def to_dict(self) -> dict:
        return self.__cfg

    def __index__(self, key: str):
        key = key.lower()
        return self.__cfg[key]
    
    def get(self, name: str, default: Optional[Any]=None):
        name = name.lower()
        if name in self.__cfg:
            return self.__cfg[name]
        else:
            return default

    def __getattr__(self, name: str):
        name = name.lower()
        if name in self.__cfg:
            return self.__cfg[name]
        else:
            raise AttributeError(f"config term {name} does not exist.")

    def __contains__(self, key: str):
        key = key.lower()
        return key in self.__cfg




def parse_args(cmd_str=None):
    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument("--name", type=str)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--force", action="store_true", default=False)

    # parser.add_argument("--e2e_test", action="store_true", default=False)
    parser.add_argument("--only_cluster", action="store_true", default=False)
    
    # resume args
    parser.add_argument("--ft_model_weight", type=str)
    parser.add_argument("--cls_models", type=str)

    # parallel args
    parser.add_argument("--local_rank", type=int, default=False)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--dist_url", type=str)
    

    if cmd_str:
        cmd_args = parser.parse_args(cmd_str.split())
    else:
        cmd_args = parser.parse_args()
    
    config = ConfigManager.init_with_yaml("config/cls/BaseConfig.yaml")
    config.override_with_yaml(cmd_args.cfg)
    
    cmd_args = vars(cmd_args)
    config.override_with_dict(cmd_args, strict=False)


    # auto-complement other fields
    if config.name is None:
        config.update("name", osp.splitext(osp.basename(config.cfg))[0])

    config.update("num_class", {
        "ImageNet_LT":  1000,
        "Places_LT":    365,
        "iNaturalist":  8142,
    }[config.dataset])
    
    if config.ft_model_weight is None:
        config.update("ft_model_weight", [])
    else:
        config.update("ft_model_weight", config.ft_model_weight.split(","))

    return config



# def source_import(file_path):
#     """This function imports python module directly from source code using importlib"""
#     spec = importlib.util.spec_from_file_location('', file_path)
#     assert spec is not None
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module

    
# def batch_show(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.figure(figsize=(20,20))
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)

def print_write(print_str, log_file):
    logging.info(" ".join(map(str, print_str)))
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(*print_str, file=f)





def prepare_log_dir(cfg: ConfigManager):
    save_dir = osp.join(cfg.root_dir, cfg.name)
    if osp.exists(save_dir):
        if cfg.force:
            for f in glob.glob(save_dir+"/*"):
                logging.warning(f"removing {f}")
                os.remove(f)
        else:
            raise FileExistsError(f"{save_dir} exists")
    else:
        os.makedirs(save_dir)
    logging.info(f"save to {save_dir}")
    shutil.copy(cfg.cfg, osp.join(save_dir, "config.yaml"))

    writer = SummaryWriter(save_dir)

    return save_dir, writer

