import argparse
import csv
import errno
import glob
import io
import os
import shutil
import oyaml
import numpy as np
import random

import torch
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from termcolor import colored
from torch import nn
from tqdm import tqdm

from modeling.SpiderNet import SpiderNet
from utils import metrics, tensorlog, lossfunc
from dataloaders.get_dataloader import get_dataloader


def train_val():
    ###################### Load Config File #############################
    parser = argparse.ArgumentParser(description='Run training of outlines prediction model')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file',
                        metavar='path/to/config')
    args = parser.parse_args()

    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH, encoding='utf-8') as fd:
        config_yaml = oyaml.safe_load(fd)

    config = AttrDict(config_yaml)

    ###################### Logs (TensorBoard)  #############################
    # Create directory to save results
    SUBDIR_RESULT = 'checkpoints'

    results_root_dir = config.train.logsDir
    runs = sorted(glob.glob(os.path.join(results_root_dir, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(os.path.join(results_dir, SUBDIR_RESULT)):
        NUM_FILES_IN_EMPTY_FOLDER = 0
        if len(os.listdir(os.path.join(results_dir, SUBDIR_RESULT))) > NUM_FILES_IN_EMPTY_FOLDER:
            prev_run_id += 1
            results_dir = os.path.join(results_root_dir, 'exp-{:03d}'.format(prev_run_id))
            os.makedirs(results_dir)
    else:
        os.makedirs(results_dir)

    try:
        os.makedirs(os.path.join(results_dir, SUBDIR_RESULT))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    MODEL_LOG_DIR = results_dir
    CHECKPOINT_DIR = os.path.join(MODEL_LOG_DIR, SUBDIR_RESULT)

    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('Saving results to folder: ' + colored('"{}"'.format(results_dir), 'blue'))

    writer = SummaryWriter(MODEL_LOG_DIR, comment='create-graph')

    string_out = io.StringIO()
    # create a yaml file
    oyaml.dump(config_yaml, string_out, default_flow_style=False)
    config_str = string_out.getvalue().split('\n')
    string = ''
    for line in config_str:
        string = string + '    ' + line + '\n\r'
    writer.add_text('Config', string, global_step=None)

    # Create CSV File to store error metrics
    csvs = {
        "syn_val": os.path.join(MODEL_LOG_DIR, 'syn_val.csv'),
        "syn_test": os.path.join(MODEL_LOG_DIR, 'syn_test.csv'),
        "real_val": os.path.join(MODEL_LOG_DIR, 'real_val.csv'),
        "real_test": os.path.join(MODEL_LOG_DIR, 'real_test.csv')
    }
    field_names = ["Epoch Num", "RMSE", "REL", "MAE", "<1.05", "<1.10", "<1.25"]
    for csv_filename in csvs.values():
        with open(csv_filename, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
            csv_writer.writeheader()

    ###################### ModelBuilder #############################
    model = SpiderNet(numStage=config.train.modelParam.numStage,
                      reduction=config.train.modelParam.reduction)
    print('The stage num in SpiderNet is: {}'.format(model.numStage))

    if config.train.continueTraining:
        print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
        if not os.path.isfile(config.train.pathPrevCheckpoint):
            raise ValueError('Invalid path to the given weights file for transfer learning.\
                       The file {} does not exist'.format(config.train.pathPrevCheckpoint))

        CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')
        model.load_state_dict(CHECKPOINT['model_state_dict'])
        model.load_state_dict(CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Enable Multi-GPU training
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    ########################### loss ################################
    criterion = lossfunc.AUDLoss(delta=1.5)

    ###################### Setup Optimizer #############################
    # optimizer
    if config.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=float(config.train.optimAdam.learningRate),
                                     weight_decay=float(config.train.optimAdam.weight_decay))
    else:
        raise ValueError(
            'Invalid optimizer "{}" in config file. Must be one of ["Adam"]'.format(config.train.optimizer))

    print("use {} optimizer".format(config.train.optimizer))

    # lr_scheduler
    if config.train.lrScheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=config.train.stepLR.step_size,
                                                       gamma=float(config.train.stepLR.gamma))
        print("use {} lr_scheduler".format(config.train.lrScheduler))
    elif config.train.lrScheduler == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=config.train.CosineAnnealingLR.T_max,
                                                                  eta_min=float(config.train.CosineAnnealingLR.eta_min))
        print("use {} lr_scheduler".format(config.train.lrScheduler))
    else:
        print(" no lr_scheduler")

    # Continue Training from prev checkpoint if required
    if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
        if 'optimizer_state_dict' in CHECKPOINT:
            optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
        else:
            print(
                colored(
                    'WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                    '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))

    if config.train.continueTraining and config.train.initLr_schedulerFromCheckpoint:
        if 'lr_scheduler_state_dict' in CHECKPOINT:
            lr_scheduler.load_state_dict(CHECKPOINT['lr_scheduler_state_dict'])
        else:
            print(
                colored(
                    'WARNING: Could not load lr_scheduler state from checkpoint as checkpoint does not contain ' +
                    '"lr_scheduler_state_dict". Continuing without loading lr_scheduler state. ', 'red'))

    ###################### Train Model #############################
    total_iter_num = 0
    START_EPOCH = 0
    END_EPOCH = config.train.numEpochs

    if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
        if 'model_state_dict' in CHECKPOINT:
            total_iter_num = CHECKPOINT['total_iter_num'] + 1
            START_EPOCH = CHECKPOINT['epoch'] + 1
            END_EPOCH = CHECKPOINT['epoch'] + config.train.numEpochs
        else:
            print(
                colored(
                    'Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.\
                           Starting from epoch num 0', 'red'))

    for epoch in range(START_EPOCH, END_EPOCH):
        print('\n\nEpoch {}/{}'.format(epoch, END_EPOCH - 1))
        print('-' * 30)

        ###################### Get DataLoader #############################
        trainLoader, valDict = get_dataloader(config)

        ###################### Training Cycle #############################
        print('Train:')
        print('=' * 20)

        model.train()

        epoch_total_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        epoch_loss3 = 0.0

        for iter_num, batch in enumerate(tqdm(trainLoader)):
            total_iter_num += 1

            # Get data
            inputs, raw_depth, labels, masks = batch

            raw_depth = raw_depth.to(device)
            labels = labels.to(device)
            inputs = inputs.to(device)
            masks = masks.to(device)

            # training
            optimizer.zero_grad()
            torch.set_grad_enabled(True)

            out1, sigma1, out2, sigma2, output_depth, sigma3 = model.forward(inputs, raw_depth)
            loss_wight = config.train.lossWeights

            loss1 = loss_wight[0] * criterion(out1, labels, sigma1)
            loss2 = loss_wight[1] * criterion(out2, labels, sigma2)
            loss3 = loss_wight[2] * criterion(output_depth, labels, sigma3)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss3 += loss3.item()

        # Update Learning Rate Scheduler
        if config.train.lrScheduler == 'StepLR' or config.train.lrScheduler == 'MultiStepLR' \
                or config.train.lrScheduler == 'CosineAnnealingLR':
            lr_scheduler.step()
        else:
            pass

        # Log Current Learning Rate  and Loss
        current_learning_rate = optimizer.param_groups[0]['lr']

        writer.add_scalar('loss/Train Epoch Loss1', epoch_loss1, epoch)
        writer.add_scalar('loss/Train Epoch Loss2', epoch_loss2, epoch)
        writer.add_scalar('loss/Train Epoch Loss3', epoch_loss3, epoch)
        writer.add_scalar('loss/Train Epoch total Loss', epoch_total_loss, epoch)

        print('loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f}'.format(epoch_loss1, epoch_loss2, epoch_loss3))
        print('Train Epoch total Loss: {:.4f}'.format(epoch_total_loss))

        writer.add_scalar('Learning Rate', current_learning_rate, epoch)


        if torch.cuda.device_count() > 1:
            model_params = model.module.state_dict()  # Saving nn.DataParallel model
        else:
            model_params = model.state_dict()

        filename = os.path.join(CHECKPOINT_DIR, 'latest.pth'.format(epoch))
        torch.save(
            {
                'model_state_dict': model_params,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'total_iter_num': total_iter_num,
                'epoch_loss': epoch_total_loss
            }, filename)

        if (epoch % config.train.saveModelInterval) == 0:
            filename = os.path.join(CHECKPOINT_DIR, 'epoch-{:04d}.pth'.format(epoch))
            torch.save(
                {
                    'model_state_dict': model_params,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'total_iter_num': total_iter_num,
                    'epoch_loss': epoch_total_loss
                }, filename)

        ###################### Validation Cycle #############################
        for valType, valLoader in valDict.items():
            print('\n' + valType + ' Validation:')
            print('=' * 20)

            model.eval()

            running_rmse = 0.0
            running_rel = 0.0
            running_mae = 0.0
            running_d105 = 0.0
            running_d110 = 0.0
            running_d125 = 0.0

            for iter_num, batch in enumerate(tqdm(valLoader)):
                # Get data
                inputs, raw_depth, labels, masks = batch
                raw_depth = raw_depth.to(device)
                labels = labels.to(device)
                inputs = inputs.to(device)
                masks = masks.to(device)

                with torch.no_grad():
                    _, _, _, _, output_depth, _ = model(inputs, raw_depth)

                rmse, abs_rel, mae, per_d105, per_d110, per_d125 = metrics.compute_errors(
                    output_depth.detach(),
                    labels.detach(),
                    masks.detach()
                )

                running_rmse += rmse
                running_rel += abs_rel
                running_mae += mae
                running_d105 += per_d105
                running_d110 += per_d110
                running_d125 += per_d125

            num_samples = len(valLoader)
            epoch_rmse = running_rmse / num_samples
            epoch_rel = running_rel / num_samples
            epoch_mae = running_mae / num_samples
            epoch_d105 = running_d105 / num_samples
            epoch_d110 = running_d110 / num_samples
            epoch_d125 = running_d125 / num_samples

            writer.add_scalar(valType + '/rmse', epoch_rmse, epoch)
            writer.add_scalar(valType + '/rel', epoch_rel, epoch)
            writer.add_scalar(valType + '/mae', epoch_mae, epoch)
            writer.add_scalar(valType + '/d105', epoch_d105, epoch)
            writer.add_scalar(valType + '/d110', epoch_d110, epoch)
            writer.add_scalar(valType + '/d125', epoch_d125, epoch)

            row_data = [epoch, round(epoch_rmse, 5), round(epoch_rel, 5), round(epoch_mae, 5),
                        round(epoch_d105, 5), round(epoch_d110, 5), round(epoch_d125, 5)]

            csv_filename = csvs.get(valType)
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                csv_writer.writerow(dict(zip(field_names, row_data)))

            print(valType + ' Epoch rmse: {:.5f}'.format(epoch_rmse))
            print(valType + ' Epoch rel: {:.5f}'.format(epoch_rel))
            print(valType + ' Epoch mae: {:.5f}'.format(epoch_mae))
            print(valType + ' Epoch d105: {:.5f}'.format(epoch_d105))
            print(valType + ' Epoch d110: {:.5f}'.format(epoch_d110))
            print(valType + ' Epoch d125: {:.5f}'.format(epoch_d125))

            if (epoch % config.train.saveImageInterval) == 0:
              rgb_img, other_image = tensorlog.create_grid_image(inputs,
                                                                 output_depth,
                                                                 raw_depth,
                                                                 labels,
                                                                 masks,
                                                                 max_num_images_to_save=1)
              writer.add_image(valType + ' input rgb', rgb_img, epoch)
              writer.add_image(valType + ' raw_depth label mask', other_image, epoch)
    writer.close()


if __name__ == '__main__':
    train_val()

