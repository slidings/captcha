# src/utils.py
import os
import re
import random
import numpy as np
import torch

def parse_label_from_name(fname: str) -> str:
    """
    Extract CAPTCHA label from filename.
    For example:
      '0a4y-0.png' → '0a4y'
    """
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]
    # split before first dash
    label = stem.split("-")[0]
    return label  # keep lowercase and digits


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
