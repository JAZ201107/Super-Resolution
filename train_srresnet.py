import time
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn

from model import SRResNet
from datasets import SRDataset
from dataclasses import dataclass

from utils import *
from configs.srresnet_config import get_config
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from tqdm import tqdm
import logging


def train(train_loader, model, criterion, optimizer, epoch, epochs, config, writer):
    device = config.DEVICE.device
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    with tqdm(total=len(train_loader)) as t:
        t.set_description(desc=f"Epoch: [{epoch}/{epochs}]", refresh=False)
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            # if i < 7703:
            #     continue
            data_time.update(time.time() - start)

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)

            loss = criterion(sr_imgs, hr_imgs)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()

            if config.LEARNING.grad_clip is not None:
                clip_gradient(optimizer, config.LEARNING.grad_clip)

            optimizer.step()

            losses.update(loss.item(), lr_imgs.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # # Print status
            if i % config.LEARNING.print_freq == 0:
                logging.info(
                    "Epoch: [{0}/{1}][{2}/{3}]----"
                    "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----"
                    "Data Time {data_time.val:.3f} ({data_time.avg:.3f})----"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                    )
                )
            writer.add_scalar(
                "Loss", losses.val, global_step=epoch * len(train_loader) + i
            )
            writer.add_scalar(
                "Loss/avg", losses.avg, global_step=epoch * len(train_loader) + i
            )
            t.set_postfix(
                loss="{:05.3f}".format(losses.val),
                loss_avg="{:05.3f}".format(losses.avg),
            )
            t.update()

    del (
        lr_imgs,
        hr_imgs,
        sr_imgs,
    )  # free some memory since their histories may be stored


def parse_args():
    parser = argparse.ArgumentParser(description="Train Super Resolution Models")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config file",
    )

    parser.add_argument(
        "--EXPERIENCE.NAME",
        type=str,
        default="srresnet",
        help="path to data folder",
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="Frequence of logging",
    )

    args = parser.parse_args()
    arg_list = []
    for arg, value in vars(args).items():
        if value is not None and arg.isupper():
            arg_list.extend([arg, str(value)])

    return args, arg_list


if __name__ == "__main__":
    # Set TensorBoard
    BASE_DIR = "./experiment/runs"

    args, arg_list = parse_args()

    # Merge Configuration
    config = get_config()
    config.merge_from_list(arg_list)
    if args.config is not None:
        config.merge_from_file(args.config)
    EXPERIMENT_NAME = os.path.join(
        BASE_DIR, f"{config.EXPERIENCE.NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(EXPERIMENT_NAME, exist_ok=True)
    save_config(config, os.path.join(EXPERIMENT_NAME, "config_srresnet.yaml"))

    writer = SummaryWriter(log_dir=os.path.join(EXPERIMENT_NAME))
    set_logger(os.path.join(EXPERIMENT_NAME, "train.log"))

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
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING.lr,
        )
        start_epoch = 0
    else:
        logging.info(
            f"Loading weight from the checkpoint {config.LEARNING.checkpoint} at {start_epoch} epoch"
        )
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        logging.info("- done.")

    model = model.to(config.DEVICE.device)
    criterion = nn.MSELoss().to(config.DEVICE.device)

    logging.info("Loading the datasets...")
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
        batch_size=config.LEARNING.batch_size,
        shuffle=True,
        num_workers=config.LEARNING.workers,
        pin_memory=True,
    )
    logging.info("- done.")

    epochs = int(config.LEARNING.iterations // len(train_loader) + 1)

    logging.info("Starting training for {} epoch(s)".format(epochs))
    for epoch in range(start_epoch, epochs):
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            epochs,
            config,
            writer,
        )
        save_checkpoint(
            {"epoch": epoch, "model": model, "optimizer": optimizer},
            os.path.join(EXPERIMENT_NAME, "checkpoint_srresnet.pth.tar"),
        )
    writer.close()
