# src/dataset.py
import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from .image_clip import resize_with_padding
from .utils import parse_label_from_name
from .vocab import encode_label
from .transforms import keep_aspect_resize_pad, to_float_tensor, basic_preprocess
from torchvision import transforms

class CaptchaDataset(Dataset):
    def __init__(self, root_dir: str, img_height: int = 32, max_width: int = 256, grayscale: bool = False,
                 is_train: bool = False): # Flag to control augmentation
        
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.*")))
        self.h = img_height
        self.max_w = max_width
        self.grayscale = grayscale
        
        self.is_train = is_train
        
        # Define the augmentation pipeline
        if self.is_train:
            self.aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
                ], p=0.7),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), shear=5)
                ], p=0.7),
                transforms.RandomApply([
                    transforms.GaussianBlur((3, 3), sigma=(0.1, 0.5))
                ], p=0.5),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.paths[idx]
        label_str = parse_label_from_name(path)
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for PIL

        if self.grayscale:
            img = basic_preprocess(img)

        # resize + pad (from transforms.py)
        img_resized, true_w = keep_aspect_resize_pad(img, self.h, self.max_w)

        # for the new resizing
        # img_resized, (_,_,true_w,_) = resize_with_padding(img, self.h, self.max_w)
        
        # --- APPLY AUGMENTATION ---
        # We must convert to PIL Image for torchvision transforms, then back
        if self.aug_transform:
            # Apply the transform pipeline
            img_aug_pil = self.aug_transform(img_resized)
            # Convert PIL image back to OpenCV (numpy) format
            img_resized = np.array(img_aug_pil) 
        # --- END AUGMENTATION ---

        # Convert to float tensor (CHW, normalized)
        tensor = to_float_tensor(img_resized)

        label_ids = encode_label(label_str)
        
        sample = {
            "image": torch.from_numpy(tensor),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "label_str": label_str,
            "true_w": true_w
        }
        return sample

def collate_fn(batch: List[Dict]) -> Dict:
    """
    - Stack images (already padded to same H and max W).
    - Concatenate targets for CTCLoss (requires 1D target with target_lengths).
    """
    images = torch.stack([b["image"] for b in batch], dim=0)   # (B,C,H,W)
    labels = [b["label_ids"] for b in batch]
    label_lengths = torch.tensor([len(x) for x in labels], dtype=torch.long)

    if any(l == 0 for l in label_lengths.tolist()):
        # edge case: empty labels after filtering — drop them or set to a dummy char
        # For now, drop empty by replacing with a zero-length tensor (CTC allows 0-length?)
        # It's cleaner to filter those samples in __getitem__ if needed.
        pass

    targets = torch.cat(labels, dim=0) if len(labels) else torch.empty(0, dtype=torch.long)

    # true_w helps estimate sequence lengths later
    true_ws = torch.tensor([b["true_w"] for b in batch], dtype=torch.long)
    label_strs = [b["label_str"] for b in batch]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": label_lengths,
        "true_ws": true_ws,
        "label_strs": label_strs
    }




