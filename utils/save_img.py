import os
import imageio
import torch
from torchvision.utils import make_grid
import numpy as np
import cv2
import errno

from utils.dataprocess import depth2rgb, exr_saver


def save_img(results_dir, subdir_imgs, index,
             img, raw_depth, output1, output2, output3, label, mask,
             min_depth=0.1, max_depth=1.5):
    # make dir
    try:
        os.makedirs(os.path.join(results_dir, subdir_imgs, 'readme_img'))
        os.makedirs(os.path.join(results_dir, subdir_imgs, 'input_depth'))

        os.makedirs(os.path.join(results_dir, subdir_imgs, 'output1'))
        os.makedirs(os.path.join(results_dir, subdir_imgs, 'output2'))
        os.makedirs(os.path.join(results_dir, subdir_imgs, 'output3'))

        os.makedirs(os.path.join(results_dir, subdir_imgs, 'gt_depth'))
        os.makedirs(os.path.join(results_dir, subdir_imgs, 'mask'))

    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # create RGB visualization of depth images
    COLOR_MAP = cv2.COLORMAP_JET
    input_depth_rgb = depth2rgb(raw_depth,
                                min_depth=min_depth,
                                max_depth=max_depth,
                                color_mode=COLOR_MAP)
    output1_rgb = depth2rgb(output1,
                            min_depth=min_depth,
                            max_depth=max_depth,
                            color_mode=COLOR_MAP)
    output2_rgb = depth2rgb(output2,
                            min_depth=min_depth,
                            max_depth=max_depth,
                            color_mode=COLOR_MAP)
    output3_rgb = depth2rgb(output3,
                            min_depth=min_depth,
                            max_depth=max_depth,
                            color_mode=COLOR_MAP)
    gt_depth_rgb = depth2rgb(label,
                             min_depth=min_depth,
                             max_depth=max_depth,
                             color_mode=COLOR_MAP)

    # Save PNG and EXR Output
    # out1
    output_path_rgb = os.path.join(results_dir, subdir_imgs,
                                   'output1/{:09d}-out1.png'.format(index))
    imageio.imwrite(output_path_rgb, output1_rgb)

    output_path_exr = os.path.join(results_dir, subdir_imgs,
                                   'output1/{:09d}-out1.exr'.format(index))
    exr_saver(output_path_exr, output1)

    # out2
    output_path_rgb = os.path.join(results_dir, subdir_imgs,
                                   'output2/{:09d}-out2.png'.format(index))
    imageio.imwrite(output_path_rgb, output2_rgb)

    output_path_exr = os.path.join(results_dir, subdir_imgs,
                                   'output2/{:09d}-out2.exr'.format(index))
    exr_saver(output_path_exr, output2)

    # out3
    output_path_rgb = os.path.join(results_dir, subdir_imgs,
                                   'output3/{:09d}-out3.png'.format(index))
    imageio.imwrite(output_path_rgb, output3_rgb)

    output_path_exr = os.path.join(results_dir, subdir_imgs,
                                   'output3/{:09d}-out3.exr'.format(index))
    exr_saver(output_path_exr, output3)

    # save input, label color image
    input_path_rgb = os.path.join(results_dir, subdir_imgs,
                             'readme_img/{:09d}-readme_img.png'.format(index))
    img = (img * 255.0).astype(np.uint8)
    imageio.imwrite(input_path_rgb, img)

    input_path_rgb = os.path.join(results_dir, subdir_imgs,
                                  'input_depth/{:09d}-in-depths.png'.format(index))
    imageio.imwrite(input_path_rgb, input_depth_rgb)

    gt_path_rgb = os.path.join(results_dir, subdir_imgs,
                               'gt_depth/{:09d}-in-depths.png'.format(index))
    imageio.imwrite(gt_path_rgb, gt_depth_rgb)

    mask_valid_pixels = (mask * 255.0).astype(np.uint8)
    # mask_valid_pixels[mask_valid_pixels > 0] = 255
    output_path_valid_mask = os.path.join(results_dir, subdir_imgs,
                                          'mask/{:09d}-depths-mask.png'.format(index))
    imageio.imwrite(output_path_valid_mask, mask_valid_pixels)


