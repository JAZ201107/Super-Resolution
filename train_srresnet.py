import time
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn

from model import SRResNet
from datasets import SRDataset
from dataclasses import dataclass

from utils import *
from .configs.srresnet_config import get_config
from torch.utils.tensorboard import SummaryWriter


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Super Resolution Models")
    parse.add_argument(
        "--CONFIG",
        type=str,
        default="./config/config.yaml",
        help="path to config file",
    )

    parse.add_argument(
        "--experiment_name",
        type=str,
        default="srresnet",
        help="path to data folder",
    )

    args = parser.parse_args()
    arg_list = []
    for arg, value in vars(args).item():
        if value is not None:
            arg_list.extend([arg, str(value)])

    return arg, arg_list


if __name__ == "__main__":
    # Set TensorBoard
    BASE_DIR = "./experiment/runs"

    arg, arg_list = parse_args()

    # Merge Configuration
    config = get_config()
    config.merge_from_list(arg_list)
    config.merge_from_file(config.CONFIG)
    EXPERIMENT_NAME = os.path.join(BASE_DIR, config.experiment_name)
    os.mkdir(EXPERIMENT_NAME)

    save_config(config, os.path.join(EXPERIMENT_NAME, "config_srresnet.yaml"))

    writer = SummaryWriter(log_dir=EXPERIMENT_NAME)

    checkpoint = config.LEARNING.checkpoint
    if checkpoint is None:
        model = SRResNet(
            large_kernel_size=config.MODEL.large_kernel_size,
            small_kernel_size=config.MODEL.small_kernel_size,
            n_channels=config.MODEL.n_channels,
            n_blocks=config.MODEL.n_blocks,
            scaling_factor=config.DATA.scaling_factor,
        )

        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
    else:
        pass

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    train_dataset = SRDataset(
        config.DATA.data_folder,
        split="train",
        crop_size=config.DATA.crop_size,
        scaling_factor=config.DATA.scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.DATA.batch_size,
        shuffle=True,
        num_workers=config.DATA.num_workers,
        pin_memory=True,
    )

    writer.close()
