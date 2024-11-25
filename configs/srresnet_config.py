from yacs.config import CfgNode as CN
import torch


__C = CN()

# ---------------------------------- DATA ------------------------------------------
__C.DATA = CN()
__C.DATA.data_folder = "/root/autodl-tmp/data"  # folder with JSON data files
__C.DATA.crop_size = 96  # crop size of target HR images
__C.DATA.scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# ---------------------------------- MODEL ------------------------------------------
__C.MODEL = CN()
__C.MODEL.large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
__C.MODEL.small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
__C.MODEL.n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
__C.MODEL.n_blocks = 16  # number of residual blocks

# ---------------------------------- LEARNING ------------------------------------------
__C.LEARNING = CN()
__C.LEARNING.checkpoint = None  # path to model checkpoint, None if none
__C.LEARNING.batch_size = 64  # batch size
__C.LEARNING.start_epoch = 0  # start at this epoch
__C.LEARNING.iterations = 1e6  # number of training iterations
__C.LEARNING.workers = 4  # number of workers for loading data in the DataLoader
__C.LEARNING.print_freq = 500  # print training status once every __ batches
__C.LEARNING.lr = 1e-4  # learning rate
__C.LEARNING.grad_clip = None  # clip if gradients are exploding

# ---------------------------------- DEFAULT DEVICE ------------------------------------------
__C.DEVICE = CN()
__C.DEVICE.device = "cuda" if torch.cuda.is_available() else "cpu"
__C.DEVICE.cudnn_benchmark = True


# ------- Experiemce ----
__C.EXPERIENCE = CN()
__C.EXPERIENCE.NAME = None


def get_config():
    return __C.clone()
