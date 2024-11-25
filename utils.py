import torch
import torch.nn as nn

import os
from PIL import Image


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, v, n=1):
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"=> Saving checkpoint: {filename}")


# Configurations
def save_config(config, save_path):
    with open(save_path, "w") as f:
        f.write(config.dump())
    print(f"=> Saving configuration file: {save_path}")
