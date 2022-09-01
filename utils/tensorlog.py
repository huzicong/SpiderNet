import torch
from torchvision.utils import make_grid


def create_grid_image(input_img, out1, input_depth, depth_label, mask, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''
    rgb_img = make_grid(input_img[:max_num_images_to_save], 1, normalize=False, scale_each=False)

    images = torch.cat((input_depth, out1, depth_label, mask), dim=3)
    other_image = make_grid(images[:max_num_images_to_save], 1, normalize=False, scale_each=False)

    return rgb_img, other_image




