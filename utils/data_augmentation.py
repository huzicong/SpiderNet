import random
import time

import numpy as np
import cv2
import torch.random

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_NORM = [0.229, 0.224, 0.225]

def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im


def add_noise(image, level=0.1):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row, col, ch = image.shape
        mean = 0
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')


def add_noise_to_depth(depth_img):
    """ Distort depth image with multiplicative gamma noise.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    gamma_noise = np.random.gamma(1000.0, 0.001)
    depth_img = depth_img * gamma_noise
    return depth_img


def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes

        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean = IMG_MEAN
    std = IMG_NORM
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized


def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor

def dropout_random_ellipses_4mask(valid_mask, ellipse_dropout_mean, ellipse_gamma_shape, ellipse_gamma_scale):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    dropout_valid_mask = valid_mask.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(ellipse_dropout_mean)

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(dropout_valid_mask > 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(ellipse_gamma_shape, ellipse_gamma_scale, size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(ellipse_gamma_shape, ellipse_gamma_scale, size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        mask = np.zeros_like(dropout_valid_mask)
        mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        dropout_valid_mask[mask == 1] = 0

    return dropout_valid_mask