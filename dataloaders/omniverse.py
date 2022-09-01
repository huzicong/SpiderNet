import cv2
import h5py
import imgaug as ia
import torch
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

# My libraries

from dataloaders.cleargrasp import CleargraspDataset
from utils import data_augmentation


class OmniverseDataset(Dataset):
    def __init__(self,
                 input_dir,
                 exp_type,
                 split_ratio=1.0,
                 transform=None,
                 input_only=None):
        self.input_dir = input_dir
        self.exp_type = exp_type
        self.split_ratio = split_ratio
        self.transform = transform
        self.input_only = input_only

        h5_paths = sorted(glob(os.path.join(self.input_dir, 'train/*/*.h5')))
        idx = int(len(h5_paths) * self.split_ratio)
        if self.exp_type == 'train':
            self.h5_paths = h5_paths[:idx]
        elif self.exp_type == 'val':
            self.h5_paths = h5_paths[idx:]
        elif self.exp_type == 'test':
            self.h5_paths = h5_paths[idx:]

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, index):
        f = h5py.File(self.h5_paths[index], "r")
        _img = f['rgb_glass'][:]
        instance_seg = f['instance_seg'][:]
        semantic_seg = f['semantic_seg'][:]
        disparity = f['depth'][:]
        f.close()

        # depth-gt-mask
        instance_id = np.arange(1, instance_seg.shape[0] + 1).reshape(-1, 1, 1)
        instance_mask = np.sum(instance_seg * instance_id, 0).astype(np.uint8)
        semantic_id = np.arange(1, semantic_seg.shape[0] + 1).reshape(-1, 1, 1)
        semantic_mask = np.sum(semantic_seg * semantic_id, 0).astype(np.uint8)
        _mask = self.get_corrupt_mask(instance_mask, semantic_mask, instance_seg.shape[0],
                                      corrupt_all=True, ratio_low=0.3, ratio_high=0.7)

        # depth, depth-gt
        _label = 1. / (disparity + 1e-8) * 0.01
        _label = np.clip(_label, 0, 4)
        _raw_depth = _label.copy()
        _raw_depth[_mask > 0] = 0

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img, hooks=ia.HooksImages(activator=self._activator_masks))

            _raw_depth = self.transform.augment_image(_raw_depth)

            _mask = det_tf.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

            _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        _img_tensor = transforms.ToTensor()(_img)

        _mask_tensor = torch.from_numpy(_mask.copy().astype(np.float32)).unsqueeze(0)

        _raw_depth_tensor = torch.from_numpy(_raw_depth.copy()).unsqueeze(0)

        _label_tensor = torch.from_numpy(_label.copy()).unsqueeze(0)

        return _img_tensor, _raw_depth_tensor, _label_tensor, _mask_tensor

    def get_corrupt_mask(self, instance_mask, semantic_mask, instance_num, corrupt_all=False, ratio_low=0.4,
                         ratio_high=0.8):
        rng = np.random.default_rng()
        corrupt_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1]))
        if self.exp_type == 'train':
            if corrupt_all:
                corrupt_obj_num = instance_num
                corrupt_obj_ids = np.arange(instance_num)
            else:
                # randomly select corrupted objects number
                corrupt_obj_num = rng.choice(np.arange(1, instance_num + 1), 1, replace=False)[0]
                # randomly select corrupted objects ids
                corrupt_obj_ids = rng.choice(instance_num, corrupt_obj_num, replace=False)
            for cur_obj_id in corrupt_obj_ids:
                cur_obj_id = cur_obj_id + 1
                nonzero_idx = np.transpose(np.nonzero(instance_mask == cur_obj_id))
                if nonzero_idx.shape[0] == 0:
                    continue
                # transparent objects: select all pixels
                if semantic_mask[nonzero_idx[0, 0], nonzero_idx[0, 1]] == 2:
                    sampled_nonzero_idx = nonzero_idx
                # opaque objects: select partial pixels.
                else:
                    ratio = np.random.random() * (ratio_high - ratio_low) + ratio_low
                    sample_num = int(nonzero_idx.shape[0] * ratio)
                    sample_start_idx = rng.choice(nonzero_idx.shape[0] - sample_num, 1, replace=False)[0]
                    sampled_nonzero_idx = nonzero_idx[sample_start_idx:sample_start_idx + sample_num]
                    # continue
                corrupt_mask[sampled_nonzero_idx[:, 0], sampled_nonzero_idx[:, 1]] = 1
        else:
            for cur_obj_id in range(instance_num):
                cur_obj_id += 1
                nonzero_idx = np.transpose(np.nonzero(instance_mask == cur_obj_id))
                if nonzero_idx.shape[0] == 0:
                    continue
                # transparent objects: select all pixels
                if semantic_mask[nonzero_idx[0, 0], nonzero_idx[0, 1]] == 2:
                    sampled_nonzero_idx = nonzero_idx
                # opaque objects: skip
                else:
                    continue
                corrupt_mask[sampled_nonzero_idx[:, 0], sampled_nonzero_idx[:, 1]] = 1

        return corrupt_mask

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


def get_train_dataloader(input_dir, train_size, test_size, augs_train,
                         input_only, augs_test, numworkers, percentagefortrain):
    cleargrasp_root_dir = os.path.join(input_dir, 'cleargrasp/')
    omniverse_root_dir = os.path.join(input_dir, 'omniverse/')
    # train dataset
    db_train = OmniverseDataset(
        input_dir=omniverse_root_dir,
        exp_type='train',
        transform=augs_train,
        input_only=input_only
    )

    train_sub = int(percentagefortrain * len(db_train))
    db_train = torch.utils.data.Subset(db_train, range(train_sub))

    print('{} training images for Omniverse dataset'.format(len(db_train)))

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

    assert (test_size <= len(db_real_val)), \
        ('validationBatchSize ({}) cannot be more than the ' +
         'number of images in validation dataset: {}').format(test_size, len(db_real_val))

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


def get_test_dataloader(input_dir, batch_size, augs_test, numworkers, percentageforVal):
    # Test Dataset
    db_test = OmniverseDataset(
        input_dir=input_dir,
        exp_type='test',
        split_ratio=percentageforVal,
        transform=augs_test,
        input_only=None
    )

    assert (batch_size <= len(db_test)), \
        ('validationBatchSize ({}) cannot be more than the ' +
         'number of images in validation dataset: {}').format(batch_size, len(db_test))

    testloader = DataLoader(db_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=numworkers,
                            drop_last=True,
                            pin_memory=True)

    return testloader

