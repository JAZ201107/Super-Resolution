import torch
import torch.nn as nn

import os
from PIL import Image
import logging


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


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # # Logging to console
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(logging.Formatter("%(message)s"))
        # logger.addHandler(stream_handler)
