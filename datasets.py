import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT

import json
import os
from PIL import Image


def convert_image(img, source, target):
    assert source in {"pil", "[0, 1]", "[-1, 1]"}, (
        "Cannot convert from source format %s!" % source
    )
    assert target in {
        "pil",
        "[0, 255]",
        "[0, 1]",
        "[-1, 1]",
        "imagenet-norm",
        "y-channel",
    }, (
        "Cannot convert to target format %s!" % target
    )

    # Convert from source to [0, 1]
    if source == "pil":
        img = FT.to_tensor(img)

    elif source == "[0, 1]":
        pass  # already in [0, 1]

    elif source == "[-1, 1]":
        img = (img + 1.0) / 2.0

    # Convert from [0, 1] to target
    if target == "pil":
        img = FT.to_pil_image(img)

    elif target == "[0, 255]":
        img = 255.0 * img

    elif target == "[0, 1]":
        pass  # already in [0, 1]

    elif target == "[-1, 1]":
        img = 2.0 * img - 1.0

    elif target == "imagenet-norm":
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == "y-channel":
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = (
            torch.matmul(255.0 * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights)
            / 255.0
            + 16.0
        )

    return img


class ImageTransforms:
    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        self.split = split.lower()

        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {"train", "test"}

    def __call__(self, img):
        # Crop
        if self.split == "train":
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize(
            (
                int(hr_img.width / self.scaling_factor),
                int(hr_img.height / self.scaling_factor),
            ),
            Image.BICUBIC,
        )

        # Sanity check
        assert (
            hr_img.width == lr_img.width * self.scaling_factor
            and hr_img.height == lr_img.height * self.scaling_factor
        )

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source="pil", target=self.lr_img_type)
        hr_img = convert_image(hr_img, source="pil", target=self.hr_img_type)

        return lr_img, hr_img


class SRDataset(Dataset):
    def __init__(
        self,
        data_folder,
        split,
        crop_size,
        scaling_factor,
        lr_img_type,
        hr_img_type,
        test_data_name=None,
    ):
        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name

        assert self.split in {"train", "test"}
        if self.split == "test" and self.test_data_name is None:
            raise ValueError("Please provide the name of the test dataset!")
        assert lr_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}
        assert hr_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}

        if self.split == "train":
            assert (
                self.crop_size % self.scaling_factor == 0
            ), "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        # Read list of image-paths
        if self.split == "train":
            with open(os.path.join(data_folder, "train_images.json"), "r") as j:
                self.images = json.load(j)
        else:
            with open(
                os.path.join(data_folder, self.test_data_name + "_test_images.json"),
                "r",
            ) as j:
                self.images = json.load(j)

        self.transform = ImageTransforms(
            split=self.split,
            crop_size=self.crop_size,
            scaling_factor=self.scaling_factor,
            lr_img_type=self.lr_img_type,
            hr_img_type=self.hr_img_type,
        )

    def __getitem__(self, i):
        img = Image.open(self.images[i], mode="r")
        img = img.convert("RGB")
        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.images)
