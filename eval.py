import argparse
import csv
import errno
import os
import glob
import shutil
import time

import imageio
from termcolor import colored
import yaml
from attrdict import AttrDict
import torch
from imgaug import augmenters as iaa
from torch import nn
from tqdm import tqdm
from torchvision.utils import make_grid
import numpy as np
import cv2

from modeling.SpiderNet import SpiderNet
from dataloaders import cleargrasp, omniverse
from utils import metrics, save_img


def test():
    print('Inference of Depth Completion model. Loading checkpoint...')

    parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
    args = parser.parse_args()

    ###################### Load Config File #############################
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH, encoding='utf-8') as fd:
        config_yaml = yaml.safe_load(fd)
    config = AttrDict(config_yaml)

    ###################### Load Checkpoint and its data #############################
    if not os.path.isfile(config.eval.pathWeightsFile):
        raise ValueError('Invalid path to the given weights file in config. The file "{}" does not exist'.format(
            config.eval.pathWeightsFile))

    # Read config file stored in the model checkpoint to re-use it's params
    CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location='cpu')
    if 'model_state_dict' in CHECKPOINT:
        print(colored('Loaded data from checkpoint {}'.format(config.eval.pathWeightsFile), 'green'))
    else:
        raise ValueError('The checkpoint file does not have model_state_dict in it.\
                             Please use the newer checkpoint files!')

    # Create directory to save results
    SUBDIR_IMGS = 'files'

    results_root_dir = config.eval.resultsDir
    runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(os.path.join(results_dir, SUBDIR_IMGS)):
        NUM_FILES_IN_EMPTY_FOLDER = 0
        if len(os.listdir(os.path.join(results_dir, SUBDIR_IMGS))) > NUM_FILES_IN_EMPTY_FOLDER:
            prev_run_id += 1
            results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
            os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)

    try:
        os.makedirs(os.path.join(results_dir, SUBDIR_IMGS))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('Saving results to folder: ' + colored('"{}"\n'.format(results_dir), 'blue'))

    # Create CSV File to store error metrics
    csv_filename = 'computed_errors_exp_{:03d}.csv'.format(prev_run_id)
    field_names = ["Image Num", "RMSE", "REL", "MAE", "<1.05", "<1.10", "<1.25"]
    with open(os.path.join(results_dir, csv_filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        writer.writeheader()

    ###################### DataLoader #############################
    augs_test = iaa.Sequential([
        iaa.Resize({
            "height": config.eval.imgHeight,
            "width": config.eval.imgWidth
        }, interpolation='nearest'),
    ])

    # Make new dataloaders for dataset
    if config.eval.dataset.type == 'cleargrasp':
        testLoader = cleargrasp.get_test_dataloader(
            input_dir=config.eval.dataset.inputDir,
            exp_type=config.eval.dataset.expType,
            obj_type=config.eval.dataset.objType,
            batch_size=config.eval.batchSize,
            augs_test=augs_test,
            numworkers=config.eval.numWorkers
        )
    else:
        raise ValueError(
            "Invalid Dataset from config file: '{}'. Valid values are ['cleargrasp']".format(
                config.eval.dataset.type))

    ###################### ModelBuilder #############################
    model = SpiderNet(numStage=config.eval.modelParam.numStage,
                      reduction=config.eval.modelParam.reduction)

    model.load_state_dict(CHECKPOINT['model_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Enable Multi-GPU training
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    ############################# Run Validation and Test Set ########################
    print('\nInference - Depth Completion')
    print('-' * 50 + '\n')
    print(colored('Results will be saved to: {}\n'.format(config.eval.resultsDir), 'green'))

    print('Running inference on {} dataset:'.format(config.eval.dataset.type))
    print('=' * 30)
    running_rmse = []
    running_rel = []
    running_mae = []
    running_d105 = []
    running_d110 = []
    running_d125 = []

    outputImgWidth = config.eval.outputImgWidth
    outputImgHeight = config.eval.outputImgHeight

    for ii, batch in enumerate(tqdm(testLoader)):
        # Get data
        inputs, raw_depth, labels, masks = batch
        inputs = inputs.to(device)
        raw_depth = raw_depth.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            # begin = time.time()
            out1, sigma1, out2, sigma2, out3, sigma3 = model(inputs, raw_depth)
            # end = time.time()
            # print(end - begin)

        # Save output images, one at a time, to results
        img_tensor = inputs.detach()
        input_depth_tensor = raw_depth.detach().squeeze(1)

        out1_tensor = out1.detach().squeeze(1)
        out2_tensor = out2.detach().squeeze(1)
        out3_tensor = out3.detach().squeeze(1)

        sigma1_tensor = sigma1.detach().squeeze(1)
        sigma2_tensor = sigma2.detach().squeeze(1)
        sigma3_tensor = sigma3.detach().squeeze(1)

        label_tensor = labels.detach().squeeze(1)
        mask_tensor = masks.squeeze(1)

        # Extract each tensor within batch and save results
        for iii, sample_batched in enumerate(zip(img_tensor, input_depth_tensor, out1_tensor, out2_tensor, out3_tensor, sigma1_tensor, sigma2_tensor, sigma3_tensor, label_tensor, mask_tensor)):
            img, raw_depth, output1, output2, output3, uncert1, uncert2, uncert3, label, mask = sample_batched

            img = cv2.resize(img.permute(1, 2, 0).cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            raw_depth = cv2.resize(raw_depth.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)

            output1 = cv2.resize(output1.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            output2 = cv2.resize(output2.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            output3 = cv2.resize(output3.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)

            uncert1 = cv2.resize(uncert1.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            uncert2 = cv2.resize(uncert2.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            uncert3 = cv2.resize(uncert3.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)

            label = cv2.resize(label.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.cpu().numpy(), (outputImgWidth, outputImgHeight), interpolation=cv2.INTER_NEAREST)

            mask[mask > 0] = 1.0

            # Calc metrics
            rmse, abs_rel, mae, per_d105, per_d110, per_d125 = metrics.compute_errors(torch.from_numpy(output3), torch.from_numpy(label), torch.from_numpy(mask))
            running_rmse.append(rmse)
            running_rel.append(abs_rel)
            running_mae.append(mae)
            running_d105.append(per_d105)
            running_d110.append(per_d110)
            running_d125.append(per_d125)

            # Write the data into a csv file
            with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                row_data = [((ii * config.eval.batchSize) + iii),
                            round(rmse, 5),
                            round(abs_rel, 5),
                            round(mae, 5),
                            round(per_d105, 5),
                            round(per_d110, 5),
                            round(per_d125, 5)]
                writer.writerow(dict(zip(field_names, row_data)))

                index = ii * config.eval.batchSize + iii
                save_img.save_img(results_dir, SUBDIR_IMGS, index,
                                  img, raw_depth, output1, output2, output3, label, mask)

    num_batches = len(testLoader)  # Num of batches
    num_images = len(testLoader.dataset)  # Num of total images
    print('\nnum_batches:', num_batches)
    print('num_images:', num_images)

    epoch_rmse = np.mean(running_rmse)
    epoch_rel = np.mean(running_rel)
    epoch_mae = np.mean(running_mae)
    epoch_d105 = np.mean(running_d105)
    epoch_d110 = np.mean(running_d110)
    epoch_d125 = np.mean(running_d125)
    print(
        '\nTest Metrics - RMSE: {:.5f}, REL: {:.5f}, MAE: {:.5f} '
        'P1: {:.5f}%, P2: {:.5f}%, p3: {:.5f}%, num_images: {}\n\n'
        .format(epoch_rmse, epoch_rel, epoch_mae, epoch_d105, epoch_d110, epoch_d125, num_images))

    # Write the mean data into a csv file
    with open(os.path.join(results_dir, csv_filename), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        row_data = ["Mean",
                    epoch_rmse,
                    epoch_rel,
                    epoch_mae,
                    epoch_d105,
                    epoch_d110,
                    epoch_d125]
        writer.writerow(dict(zip(field_names, row_data)))


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test()

