import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# My libraries
from dataloaders.cleargrasp import CleargraspDataset
from dataloaders.omniverse import OmniverseDataset


class MixedDataset(Dataset):
    def __init__(self,
                 cleargrasp_root_dir,
                 omniverse_root_dir,
                 exp_type,
                 split_ratio,
                 transform=None,
                 input_only=None):
        self.exp_type = exp_type
        self.cleargrasp_syn_dataset = CleargraspDataset(cleargrasp_root_dir,
                                                        exp_type='train', obj_type='syn',
                                                        transform=transform,
                                                        input_only=input_only)
        self.omniverse_dataset = OmniverseDataset(omniverse_root_dir,
                                                  exp_type='train', split_ratio=split_ratio,
                                                  transform=transform,
                                                  input_only=input_only)
        self.cleargrasp_syn_len = self.cleargrasp_syn_dataset.__len__()
        self.omniverse_len = self.omniverse_dataset.__len__()

    def __getitem__(self, idx):
        if idx < self.cleargrasp_syn_len:
            return self.cleargrasp_syn_dataset.__getitem__(idx)
        else:
            return self.omniverse_dataset.__getitem__(idx-self.cleargrasp_syn_len)

    def __len__(self):
        return self.cleargrasp_syn_len + self.omniverse_len


def get_train_dataloader(input_dir, train_size, test_size, augs_train,
                         input_only, augs_test, numworkers, split_ratio, percentagefortrain):
    # train dataset
    cleargrasp_root_dir = os.path.join(input_dir, 'cleargrasp/')
    omniverse_root_dir = os.path.join(input_dir, 'omniverse/')
    db_train = MixedDataset(
        cleargrasp_root_dir=cleargrasp_root_dir,
        omniverse_root_dir=omniverse_root_dir,
        exp_type='train',
        split_ratio=split_ratio,
        transform=augs_train,
        input_only=input_only
    )

    train_sub = int(percentagefortrain * len(db_train))
    db_train = torch.utils.data.Subset(db_train, range(train_sub))

    print('{} training images for mixed dataset'.format(len(db_train)))

    # Validation Dataset
    # val syn
    db_syn_val = CleargraspDataset(
        input_dir=cleargrasp_root_dir,
        exp_type='val',
        obj_type='syn',
        transform=augs_test,
        input_only=None
    )

    db_syn_test = CleargraspDataset(
        input_dir=cleargrasp_root_dir,
        exp_type='test',
        obj_type='syn',
        transform=augs_test,
        input_only=None
    )

    # val real
    db_real_val = CleargraspDataset(
        input_dir=cleargrasp_root_dir,
        exp_type='val',
        obj_type='real',
        transform=augs_test,
        input_only=None
    )

    db_real_test = CleargraspDataset(
        input_dir=cleargrasp_root_dir,
        exp_type='test',
        obj_type='real',
        transform=augs_test,
        input_only=None
    )

    print('{} validation images for ClearGrasp syn-val dataset'.format(len(db_syn_val)))
    print('{} validation images for ClearGrasp syn-test dataset'.format(len(db_syn_test)))
    print('{} validation images for ClearGrasp real-val dataset'.format(len(db_real_val)))
    print('{} validation images for ClearGrasp real-test dataset'.format(len(db_real_test)))


    trainloader = DataLoader(
        db_train,
        batch_size=train_size,
        shuffle=True,
        num_workers=numworkers,
        drop_last=True,
        pin_memory=True,
        # prefetch_factor=6,
        persistent_workers=True,
    )

    synvalloader = DataLoader(db_syn_val,
                              batch_size=test_size,
                              num_workers=numworkers,
                              drop_last=True,
                              shuffle=False,
                              pin_memory=True)

    syntestloader = DataLoader(db_syn_test,
                               batch_size=test_size,
                               num_workers=numworkers,
                               drop_last=True,
                               shuffle=False,
                               pin_memory=True)

    realvalloader = DataLoader(db_real_val,
                               batch_size=test_size,
                               num_workers=numworkers,
                               drop_last=True,
                               shuffle=False,
                               pin_memory=True)

    realtestloader = DataLoader(db_real_test,
                                batch_size=test_size,
                                num_workers=numworkers,
                                drop_last=True,
                                shuffle=False,
                                pin_memory=True)

    valDict = {
        "syn_val": synvalloader,
        "syn_test": syntestloader,
        "real_val": realvalloader,
        "real_test": realtestloader
    }

    return trainloader, valDict

