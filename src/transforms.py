
import cv2
import numpy as np

def keep_aspect_resize_pad(img, target_h=32, max_w=128):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    new_w = min(new_w, max_w)
    img_resized = cv2.resize(img, (new_w, target_h))
    pad_w = max_w - new_w
    img_padded = cv2.copyMakeBorder(img_resized, 0, 0, 0, pad_w,
                                    cv2.BORDER_CONSTANT, value=(255,255,255))
    return img_padded, new_w

def basic_preprocess(img):
    """
    Basic denoising + binarization for CAPTCHA.
    Converts to grayscale, smooths, and thresholds.
    """
    # convert to grayscale if RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # small blur to remove line noise
    img = cv2.medianBlur(img, 3)

    # adaptive or Otsu threshold (makes text stand out)
    img = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # back to 3 channels for consistency (CNN expects 3)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def to_float_tensor(img):
    """
    Convert image to float tensor-style numpy array (CHW, normalized).
    """
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img - mean) / std
    return np.transpose(img, (2, 0, 1))
