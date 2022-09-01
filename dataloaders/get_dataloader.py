import random

import numpy as np

from dataloaders import cleargrasp, mixed, omniverse
from imgaug import augmenters as iaa
import torchvision.transforms.transforms


###################### DataLoader #############################
def get_dataloader(config):
    # imgaug
    augs_train = iaa.Sequential([
        # Geometric Augs
        iaa.Resize({
            "height": config.train.imgHeight,
            "width": config.train.imgWidth
        }, interpolation='nearest'),
    ])

    input_only = [
        "cdrop", "drop"
    ]

    augs_test = iaa.Sequential([
        iaa.Resize({
            "height": config.train.imgHeight,
            "width": config.train.imgWidth
        }, interpolation='nearest'),
    ])

    # get datasets cleargrasp
    if config.train.dataset.type == 'cleargrasp':
        trainLoader, valDict = cleargrasp.get_train_dataloader(input_dir=config.train.dataset.inputDir,
                                                               train_size=config.train.batchSize,
                                                               test_size=config.train.valBatchSize,
                                                               augs_train=augs_train,
                                                               input_only=input_only,
                                                               augs_test=augs_test,
                                                               numworkers=config.train.numWorkers,
                                                               percentagefortrain=config.train.percentageDataForTraining)
    elif config.train.dataset.type == 'omniverse':
        trainLoader, valDict = omniverse.get_train_dataloader(input_dir=config.train.dataset.inputDir,
                                                              train_size=config.train.batchSize,
                                                              test_size=config.train.valBatchSize,
                                                              augs_train=augs_train,
                                                              input_only=input_only,
                                                              augs_test=augs_test,
                                                              numworkers=config.train.numWorkers,
                                                              percentagefortrain=config.train.percentageDataForTraining)
    elif config.train.dataset.type == 'mixed':
        trainLoader, valDict = mixed.get_train_dataloader(input_dir=config.train.dataset.inputDir,
                                                          train_size=config.train.batchSize,
                                                          test_size=config.train.valBatchSize,
                                                          augs_train=augs_train,
                                                          input_only=input_only,
                                                          augs_test=augs_test,
                                                          numworkers=config.train.numWorkers,
                                                          split_ratio=config.train.dataset.omni_split_ratio,
                                                          percentagefortrain=config.train.percentageDataForTraining)
    else:
        raise ValueError(
            "Invalid Dataset from mask_config file: '{}'. Valid values are ['cleargrasp', 'omniverse', 'mixed']".format(
                config.train.dataset.type))

    return trainLoader, valDict
