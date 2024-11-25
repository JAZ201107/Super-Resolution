from .configs.srgan_config import get_config
from utils import *

from model import Generator, Discriminator, TruncatedVGG19


def train(
    train_loader,
    generator,
    discriminator,
    truncated_vgg19,
    content_loss_criterion,
    adversarial_loss_criterion,
    optimizer_g,
    optimizer_d,
    epoch,
):
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        sr_imgs = convert_image(
            sr_imgs, source="[-1, 1]", target="imagenet-norm"
        )  # (N, 3, 96, 96), imagenet-normed

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(
            hr_imgs
        ).detach()  # detached because they're constant, targets

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(
            sr_imgs_in_vgg_space, hr_imgs_in_vgg_space
        )
        adversarial_loss = adversarial_loss_criterion(
            sr_discriminated, torch.ones_like(sr_discriminated)
        )
        perceptual_loss = content_loss + beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
        # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(
            sr_discriminated, torch.zeros_like(sr_discriminated)
        ) + adversarial_loss_criterion(
            hr_discriminated, torch.ones_like(hr_discriminated)
        )

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        # Keep track of batch times
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]----"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----"
                "Data Time {data_time.val:.3f} ({data_time.avg:.3f})----"
                "Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----"
                "Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----"
                "Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss_c=losses_c,
                    loss_a=losses_a,
                    loss_d=losses_d,
                )
            )

    del (
        lr_imgs,
        hr_imgs,
        sr_imgs,
        hr_imgs_in_vgg_space,
        sr_imgs_in_vgg_space,
        hr_discriminated,
        sr_discriminated,
    )  # free some memory since their histories may be stored


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRGAN")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="path to model (SRGAN) checkpoint"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Handle Configuration
    config = get_config()
    args, args_list = parse_args()
    config.merge_from_file(args.CONFIG)
    config.merge_from_list(args_list)
    save_config(config, os.path.join(config.checkpoint, "srgan_config.yaml"))

    checkpoint = config.LEARNING.checkpoint
    if checkpoint is None:
        generator = Generator(
            large_kernel_size=config.GENERATOR.large_kernel_size_g,
            small_kernel_size=config.GENERATOR.small_kernel_size_g,
            n_channels=config.GENERATOR.n_channels_g,
            n_blocks=config.GENERATOR.n_blocks_g,
            scaling_factor=config.DATA.scaling_factor,
        )
        generator.load_weight(config.GENERATOR.srresnet_checkpoint)

        optimizer_g = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, generator.parameters()),
            lr=config.LEARNING.lr,
        )

        discriminator = Discriminator(
            kernel_size=config.DISCRIMINATOR.kernel_size_d,
            n_channels=config.DISCRIMINATOR.n_channels_d,
            n_blocks=config.DISCRIMINATOR.n_blocks_d,
            fc_size=config.DISCRIMINATOR.fc_size_d,
        )

        optimizer_d = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, discriminator.parameters()),
            lr=config.LEARNING.lr,
        )
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        generator = checkpoint["generator"]
        discriminator = checkpoint["discriminator"]
        optimizer_g = checkpoint["optimizer_g"]
        optimizer_d = checkpoint["optimizer_d"]
        print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint["epoch"] + 1))

    truncated_vgg19 = TruncatedVGG19(
        i=config.LEARNING.vgg19_i, j=config.LEARNING.vgg19_j
    )
    truncated_vgg19.eval()

    # Loss Function
    content_loss_criterion = nn.MSELoss
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # Move to default device
    generator = generator.to(config.DEVICE.device)
    discriminator = discriminator.to(config.DEVICE.device)
    truncated_vgg19 = truncated_vgg19.to(config.DEVICE.device)
    content_loss_criterion = content_loss_criterion.to(config.DEVICE.device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(config.DEVICE.device)

    # Custom dataloaders
    train_dataset = SRDataset(
        config.DATA.data_folder,
        split="train",
        crop_size=config.DATA.crop_size,
        scaling_factor=config.DATA.scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="imagenet-norm",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.LEARNING.batch_size,
        shuffle=True,
        num_workers=config.LEARNING.workers,
        pin_memory=True,
    )

    # Total number of epochs to train for
    epochs = int(config.LEARNING.iterations // len(train_loader) + 1)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # At the halfway point, reduce learning rate to a tenth
        if epoch == int((config.LEARNING.iterations / 2) // len(train_loader) + 1):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # One epoch's training
        train(
            train_loader=train_loader,
            generator=generator,
            discriminator=discriminator,
            truncated_vgg19=truncated_vgg19,
            content_loss_criterion=content_loss_criterion,
            adversarial_loss_criterion=adversarial_loss_criterion,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            epoch=epoch,
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "generator": generator,
                "discriminator": discriminator,
                "optimizer_g": optimizer_g,
                "optimizer_d": optimizer_d,
            },
            os.path.join(config.checkpoint, f"latest_checkpoint.pth.tar"),
        )
