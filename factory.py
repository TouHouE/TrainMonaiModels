from typing import Dict
import json
from torch import nn
from torch import optim

from monai.data import DataLoader, CacheDataset, PersistentDataset
from monai.networks.nets import UNet

from utils.data import change2monai_ds, get_aug


def get_optimizer(model: nn.Module, config):
    Initializer: optim.Optimizer
    ocfg = config['optim']

    if ocfg['name'] == 'adamw':
        Initializer = optim.AdamW
    elif ocfg['name'] == 'adam':
        Initializer = optim.Adam
    elif ocfg['name'] == 'sgd':
        Initializer = optim.SGD

    return Initializer(model.parameters(), **ocfg['param'])


def get_model(config) -> nn.Module:
    Initializer: nn.Module = None
    mcfg = config['model']

    if mcfg['name'] == 'unet':
        Initializer = UNet

    return Initializer(**mcfg['param'])


def get_loader(config) -> Dict[str, DataLoader]:
    dcfg = config['data']
    lcfg = config['loader']
    root = dcfg['image_root']
    with open(dcfg['json_path'], 'r') as jin:
        table = json.load(jin)
    all_ds = change2monai_ds(table, root, dcfg['fold'])
    loader_map = dict()
    trans = get_aug()

    for key, _ds in all_ds.items():
        ds = PersistentDataset(_ds, transform=trans, cache_dir=f'/workspace/{key}_cache')
        # ds = CacheDataset(_ds, transform=trans, cache_rate=1)
        print(f'# of sample in {key}: {len(ds)}')
        loader_map[key] = DataLoader(ds, **lcfg[key])
    return loader_map


