import torch
import torch.nn as nn
import torchvision
import math


# Padding = kernel_size // 2 to keep the spatial dimensions the same
class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        batch_norm=False,
        activation=None,
    ):
        super().__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {"prelu", "leakyrelu", "tanh"}

        layers = list()

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        )

        if batch_norm is not None:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation == "prelu":
            layers.append(nn.PReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2))
        elif activation == "tanh":
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class SubPixelConvolutionalBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * (scaling_factor**2),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super().__init__()

        self.conv1 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation="PReLU",
        )

        self.conv2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation=None,
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        out += residual

        return out


class SRResNet(nn.Module):
    def __init__(
        self,
        large_kernel_size=9,
        small_kernel_size=3,
        n_channels=64,
        n_blocks=16,
        scaling_factor=4,
    ):
        super().__init__()

        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # First Convolution Block
        self.conv_block1 = ConvolutionalBlock(
            in_channels=3,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="PReLU",
        )

        # A series of B Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    kernel_size=small_kernel_size,
                    n_channels=n_channels,
                )
                for _ in range(n_blocks)
            ]
        )

        # Another Convolution Block
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=True,
            activation=None,
        )

        # Upsampling
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))

        self.subpixel_convolutional_blocks = nn.Sequential(
            *[
                SubPixelConvolutionalBlock(
                    kernel_size=small_kernel_size,
                    n_channels=n_channels,
                    scaling_factor=2,
                )
                for i in range(n_subpixel_convolution_blocks)
            ]
        )

        # Final Convolution Block
        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="Tanh",
        )

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
        residual = output  # (N, n_channels, w, h)
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)
        output = self.subpixel_convolutional_blocks(
            output
        )  # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(
            output
        )  # (N, 3, w * scaling factor, h * scaling factor)

        return sr_imgs


class Generator(nn.Module):
    def __init__(
        self,
        large_kernel_size=9,
        small_kernel_size=3,
        n_channels=64,
        n_blocks=16,
        scaling_factor=4,
    ):
        super().__init__()

        self.net = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )

    def load_weight(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["state_dict"])

    def forward(self, lr_imgs):
        sr_imgs = self.net(lr_imgs)
        return sr_imgs


class Discriminator(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        n_channels=64,
        n_blocks=8,
        fc_size=1024,
    ):
        super().__init__()

        in_channels = 3

        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (
                (n_channels if i == 0 else in_channels * 2)
                if (i % 2) == 0
                else in_channels
            )

            conv_blocks.append(
                ConvolutionalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i == 0 else 2,
                    batch_norm=True if i != 0 else False,
                    activation="LeakyReLU",
                )
            )
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)

        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):

    def __init__(self, i, j):

        super().__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0

        for layer in vgg19.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            if maxpool_counter == i - 1 and conv_counter == j:
                break

        assert (
            maxpool_counter == i - 1 and conv_counter == j
        ), "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (i, j)

        self.truncated_vgg19 = nn.Sequential(
            *list(vgg19.features.children())[: truncate_at + 1]
        )

    def forward(self, input):

        output = self.truncated_vgg19(input)
        return output
