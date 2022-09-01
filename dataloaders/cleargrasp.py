import imageio
import imgaug as ia
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# My libraries
from utils.dataprocess import exr_loader
from utils import data_augmentation


class CleargraspDataset(Dataset):
    def __init__(
            self,
            input_dir,
            exp_type,
            obj_type,
            transform=None,
            input_only=None
    ):
        super(CleargraspDataset, self).__init__()
        self.input_dir = input_dir
        self.exp_type = exp_type
        self.obj_type = obj_type
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input_img = []
        self._datalist_input_depth = []
        self._datalist_depth = []
        self._datalist_mask = []
        self._create_lists_filenames(self.input_dir)
        self._datalist_input_img = sorted(self._datalist_input_img)
        self._datalist_input_depth = sorted(self._datalist_input_depth)
        self._datalist_depth = sorted(self._datalist_depth)
        self._datalist_mask = sorted(self._datalist_mask)

    def __len__(self):
        return len(self._datalist_input_img)

    def __getitem__(self, index):
        # Open input imgs
        image_path = self._datalist_input_img[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)
        # _img = cv2.imread(image_path)
        # if self.exp_type == "train" and np.random.random() > 0.2:
        #     _img = data_augmentation.chromatic_transform(_img)
        #     _img = data_augmentation.add_noise(_img)

        # Open labels
        label_path = self._datalist_depth[index]
        _label = exr_loader(label_path, ndim=1)  # (H, W)
        _label[np.isnan(_label)] = 0
        _label[np.isinf(_label)] = 0

        # open masks
        mask_path = self._datalist_mask[index]
        _mask = np.array(imageio.v2.imread(mask_path))

        # open raw depth
        if len(self._datalist_input_depth) > 0:
            raw_depth_path = self._datalist_input_depth[index]
            _raw_depth = exr_loader(raw_depth_path, ndim=1)  # (H, W)
            _raw_depth[np.isnan(_raw_depth)] = 0
            _raw_depth[np.isinf(_raw_depth)] = 0
        else:
            _raw_depth = exr_loader(label_path, ndim=1)
            _raw_depth[np.isnan(_raw_depth)] = 0
            _raw_depth[np.isinf(_raw_depth)] = 0
            _raw_depth[_mask > 0] = 0

        # Apply image augmentations and convert to Tensor
        if self.transform:
            # 将这个增广数从随机的变为确定性的
            _img = self.transform.augment_image(_img, hooks=ia.HooksImages(activator=self._activator_masks))

            _raw_depth = self.transform.augment_image(_raw_depth)

            _mask = self.transform.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

            _label = self.transform.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        _mask_tensor = torch.from_numpy(_mask.copy().astype(np.float32)).unsqueeze(0)
        _mask_tensor = _mask_tensor / 255

        _raw_depth_tensor = torch.from_numpy(_raw_depth.copy()).unsqueeze(0)

        _label_tensor = torch.from_numpy(_label.copy()).unsqueeze(0)

        return _img_tensor, _raw_depth_tensor, _label_tensor, _mask_tensor

    def _create_lists_filenames(self, input_dir):
        """创建遍历的文件列表"""
        if self.exp_type == 'train' or self.obj_type == 'syn':
            if self.exp_type == 'train':
                root_dir = os.path.join(input_dir, 'cleargrasp-dataset-train/')
                subs_dir = ['cup-with-waves-train/', 'flower-bath-bomb-train/', 'heart-bath-bomb-train/',
                          'square-plastic-bottle-train/', 'stemless-plastic-champagne-glass-train/']
            elif self.exp_type == 'val':
                root_dir = os.path.join(input_dir, 'cleargrasp-dataset-test-val/', 'synthetic-val/')
                subs_dir = ['cup-with-waves-val/', 'flower-bath-bomb-val/', 'heart-bath-bomb-val/',
                            'square-plastic-bottle-val/', 'stemless-plastic-champagne-glass-val/']
            elif self.exp_type == 'test':
                root_dir = os.path.join(input_dir, 'cleargrasp-dataset-test-val/', 'synthetic-test/')
                subs_dir = ['glass-round-potion-test/', 'glass-square-potion-test/', 'star-bath-bomb-test/',
                            'tree-bath-bomb-test/']
            else:
                raise ValueError("the exp_type should be one of them: ['train', 'val', 'test']")

            for subdir in subs_dir:
                listdir = os.path.join(root_dir, subdir)
                cur_img_paths = sorted( glob(os.path.join(listdir, 'rgb-imgs/', '*-rgb.jpg')) )

                cur_mask_paths = [p.replace('rgb-imgs', 'segmentation-masks').replace('-rgb.jpg', '-segmentation-mask.png') for p in cur_img_paths]
                cur_depth_paths = [p.replace('rgb-imgs', 'depth-imgs-rectified').replace('-rgb.jpg', '-depth-rectified.exr') for p in cur_img_paths]

                self._datalist_input_img += cur_img_paths
                self._datalist_depth += cur_depth_paths
                self._datalist_mask += cur_mask_paths
            # print(self._datalist_depth)

        elif self.obj_type == 'real':
            if self.exp_type == 'val':
                root_dir = os.path.join(input_dir, 'cleargrasp-dataset-test-val/', 'real-val/')
                subs_dir = ['d435/']
            elif self.exp_type == 'test':
                root_dir = os.path.join(input_dir, 'cleargrasp-dataset-test-val/', 'real-test/')
                subs_dir = ['d415/', 'd435/']
            else:
                raise ValueError("the exp_type should be one of them: ['train', 'val', 'test']")

            for subdir in subs_dir:
                listdir = os.path.join(root_dir, subdir)
                cur_img_paths = sorted( glob(os.path.join(listdir, '*-transparent-rgb-readme_img.jpg')) )
                cur_raw_depth_paths = [p.replace('-transparent-rgb-readme_img.jpg', '-transparent-depth-readme_img.exr') for p in cur_img_paths]
                cur_mask_paths = [p.replace('-transparent-rgb-readme_img.jpg', '-mask.png') for p in cur_img_paths]
                cur_depth_paths = [p.replace('-transparent-rgb-readme_img.jpg', '-opaque-depth-readme_img.exr') for p in cur_img_paths]

                self._datalist_input_img += cur_img_paths
                self._datalist_depth += cur_depth_paths
                self._datalist_input_depth += cur_raw_depth_paths
                self._datalist_mask += cur_mask_paths
            # print(self._datalist_input_depth)

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


def get_train_dataloader(input_dir, train_size, test_size, augs_train, input_only,
                         augs_test, numworkers, percentagefortrain):

    cleargrasp_root_dir = os.path.join(input_dir, 'cleargrasp/')

    # train dataset
    db_train = CleargraspDataset(
        input_dir=cleargrasp_root_dir,
        exp_type='train',
        obj_type='syn',
        transform=augs_train,
        input_only=input_only
    )

    train_sub = int(percentagefortrain * len(db_train))
    db_train = torch.utils.data.Subset(db_train, range(train_sub))

    print('{} training images for ClearGrasp dataset'.format(len(db_train)))

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
        # prefetch_factor=16,
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


def get_test_dataloader(input_dir, exp_type, obj_type, batch_size, augs_test, numworkers):
    cleargrasp_root_dir = os.path.join(input_dir, 'cleargrasp/')
    db_test = CleargraspDataset(
        input_dir=cleargrasp_root_dir,
        exp_type=exp_type,
        obj_type=obj_type,
        transform=augs_test,
        input_only=None
    )

    assert (batch_size <= len(db_test)), \
        ('testBatchSize ({}) cannot be more than the ' +
         'number of images in test dataset: {}').format(batch_size, len(db_test))

    testloader = DataLoader(db_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=numworkers,
                            drop_last=True,
                            pin_memory=True)

    return testloader
