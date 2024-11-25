from yacs.config import CfgNode as CN


__C = CN()


# ---------------------------------- DATA ------------------------------------------
__C.DATA = CN()
__C.DATA.data_folder = "./data"  # folder with JSON data files
__C.DATA.crop_size = 96  # crop size of target HR images
__C.DATA.scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# ---------------------------------- GENERATOR ------------------------------------------
__C.GENERATOR = CN()
__C.GENERATOR.large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
__C.GENERATOR.small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
__C.GENERATOR.n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
__C.GENERATOR.n_blocks_g = 16  # number of residual blocks
__C.GENERATOR.srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"  # filepath of the trained SRResNet checkpoint used for initialization

# ---------------------------------- DISCRIMINATOR ------------------------------------------
__C.DISCRIMINATOR = CN()
__C.DISCRIMINATOR.kernel_size_d = 3  # kernel size in all convolutional blocks
__C.DISCRIMINATOR.n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
__C.DISCRIMINATOR.n_blocks_d = 8  # number of convolutional blocks
__C.DISCRIMINATOR.fc_size_d = 1024  # size of the first fully connected layer

# ---------------------------------- LEARNING ------------------------------------------
__C.LEARNING = CN()
__C.LEARNING.checkpoint = None  # path to model (SRGAN) checkpoint, None if none
__C.LEARNING.batch_size = 16  # batch size
__C.LEARNING.start_epoch = 0  # start at this epoch
__C.LEARNING.iterations = 2e5  # number of training iterations
__C.LEARNING.workers = 4  # number of workers for loading data in the DataLoader
__C.LEARNING.vgg19_i = (
    5  # the index i in the definition for VGG loss; see paper or models.py
)
__C.LEARNING.vgg19_j = (
    4  # the index j in the definition for VGG loss; see paper or models.py
)
__C.LEARNING.beta = (
    1e-3  # the coefficient to weight the adversarial loss in the perceptual loss
)
__C.LEARNING.print_freq = 500  # print training status once every __ batches
__C.LEARNING.lr = 1e-4  # learning rate
__C.LEARNING.grad_clip = None  # clip if gradients are exploding

# ---------------------------------- DEFAULT DEVICE ------------------------------------------
__C.DEVICE = CN()
__C.DEVICE.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.DEVICE.cudnn.benchmark = True


def get_config():
    return __C.clone()
