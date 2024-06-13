import os
import monai
from monai import transforms as MF


def change2monai_ds(image_label_pair_totalsegmentator_style: dict, root: str, fold: int, return_dict: bool=True):
    ts_style_ds = image_label_pair_totalsegmentator_style.copy()
    train_ds, val_ds, test_ds = dict(image=[], label=[]), dict(image=[], label=[]), dict(image=[], label=[])

    train_pack = ts_style_ds['training']
    test_pack = ts_style_ds['testing']

    for pack in train_pack:
        if pack['fold'] == fold:
            val_ds['image'].append(os.path.join(root, pack['image']))
            val_ds['label'].append(os.path.join(root, pack['label']))
        else:
            train_ds['image'].append(os.path.join(root, pack['image']))
            train_ds['label'].append(os.path.join(root, pack['label']))
    for pack in test_pack:
        test_ds['image'].append(os.path.join(root, pack['image']))
        test_ds['label'].append(os.path.join(root, pack['label']))
    if return_dict:
        return {
            "train": train_ds.copy(),
            'val': val_ds.copy(),
            'test': test_ds.copy()
        }
    return (train_ds, val_ds, test_ds)


def get_aug(phase='train', custom_layer=None):
    comp = [
        MF.LoadImaged(keys=['image', 'label']),
        MF.EnsureChannelFirstd(keys=['image', 'label']),
        MF.Spacingd(keys=['image', 'label'], pixdim=(1, 1, 1)),
        MF.Resized(keys=['image', 'label'], spatial_size=(512, 512, 320), mode=['trilinear', 'area'])
    ]

    comp.extend([
        MF.ScaleIntensityRanged(keys=['image'], a_min=-1024, a_max=1024, b_min=-1, b_max=1, clip=True)
    ])
    return MF.Compose(comp)


