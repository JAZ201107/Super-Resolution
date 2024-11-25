import time
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn

from model import SRResNet
from datasets import SRDataset
from dataclasses import dataclass

from utils import *


@dataclass
class SRResnetConfig:
    data_folder = "./"
    crop_size = 96
    scaling_factor = 4

    # Model Parameters
    large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
    small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
    n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
    n_blocks = 16  # number of residual blocks

    # Learning parameters
    checkpoint = None  # path to model checkpoint, None if none
    batch_size = 16  # batch size
    start_epoch = 0  # start at this epoch
    iterations = 1e6  # number of training iterations
    workers = 4  # number of workers for loading data in the DataLoader
    print_freq = 500  # print training status once every __ batches
    lr = 1e-4  # learning rate
    grad_clip = None  # clip if gradients are exploding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True


def main():
    if checkpoint is None:
        model = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )

        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
    else:
        pass

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    train_dataset = SRDataset(
        data_folder,
        split="train",
        crop_size=crop_size,
        scaling_factor=scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        src_imgs = model(lr_imgs)

        loss = criterion(src_imgs, hr_imgs)

        # Optimizer
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item(), lr_imgs.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]----"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----"
                "Data Time {data_time.val:.3f} ({data_time.avg:.3f})----"
                "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
    del (
        lr_imgs,
        hr_imgs,
        sr_imgs,
    )  # free some memory since their histories may be stored


if __name__ == "__main__":
    main()
