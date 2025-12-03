import cv2
import numpy as np
import glob
import os
import yaml

def add_edge_channel(img_rgb: np.ndarray) -> np.ndarray:
    # img_rgb: (H,W,3), uint8
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # normalize to [0,255]
    if mag.max() > 0:
        mag = mag / mag.max()
    mag = (mag * 255.0).astype(np.uint8)

    edge = mag[..., None]                  # (H,W,1)
    img4 = np.concatenate([img_rgb, edge], axis=2)  # (H,W,4)
    return img4

def keep_aspect_resize_pad(img, target_h=32, max_w=128):
    """
    Improved version (NO CROPPING):
    - Removes black noise lines
    - Boosts contrast for pale coloured characters using LAB + CLAHE
    - DOES NOT CROP ANYTHING
    - Resizes whole image with aspect ratio preserved
    - Pads to (target_h, max_w)
    
    img: RGB image (as in your dataset pipeline)
    returns: (canvas, new_w) where
        canvas: (target_h, max_w, 3) uint8
        new_w: width of the resized image before padding
    """

    # -----------------------------
    # 1. Remove black noise lines
    # -----------------------------
    # Black-ish lines (0,0,0) ~ (40,40,40)
    mask_black = cv2.inRange(img, (0, 0, 0), (40, 40, 40))
    img_clean = cv2.inpaint(img, mask_black, 3, cv2.INPAINT_NS)

    # -----------------------------
    # 2. Convert to LAB & enhance pale colours
    # -----------------------------
    lab = cv2.cvtColor(img_clean, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # combine A+B to make faint colours pop
    ab_combo = cv2.addWeighted(A, 0.5, B, 0.5, 0)

    # local contrast enhancement (critical for pale digits)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(ab_combo)

    # we do NOT use enh for cropping anymore, but we *could*
    # use it later if you want a mask for something else

    # -----------------------------
    # 3. Resize whole image with aspect ratio
    # -----------------------------
    h, w = img_clean.shape[:2]

    scale = min(target_h / h, max_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # choose interpolation based on whether we're shrinking or enlarging
    if scale < 1.0:
        interp = cv2.INTER_AREA      # downscale → anti-alias
    else:
        interp = cv2.INTER_CUBIC     # upscale → sharper

    resized = cv2.resize(img_clean, (new_w, new_h), interpolation=interp)

    # -----------------------------
    # 4. Pad to target size (centered)
    # -----------------------------
    canvas = np.full((target_h, max_w, 3), 255, dtype=np.uint8)

    y0 = (target_h - new_h) // 2
    x0 = (max_w - new_w) // 2

    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    return canvas, new_w

# This is mainly for converting colour to grayscale
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
    Convert image to float tensor-style numpy array (CHW).
    Automatically handles 3-channel RGB or 4-channel RGB+edge.
    """

    img = img.astype(np.float32) / 255.0

    C = img.shape[2]  # detect channels dynamically

    # Build mean/std per channel
    mean = np.array([0.5] * C, dtype=np.float32)
    std  = np.array([0.5] * C, dtype=np.float32)

    # Normalize
    img = (img - mean) / std

    return np.transpose(img, (2, 0, 1))

def visualize_crop(image_path, target_h, max_w):
    """
    Loads an image, runs keep_aspect_resize_pad,
    and displays original + final padded image.

    target_h, max_w directly control the output size.
    """
    # 1. Load
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: Could not read image at {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Apply your resize+pad
    final_rgb, new_w = keep_aspect_resize_pad(img_rgb, target_h=target_h, max_w=max_w)

    print(f"\nFile: {os.path.basename(image_path)}")
    print(f"  original shape: {img_rgb.shape}  (H, W, C)")
    print(f"  final   shape:  {final_rgb.shape} (H, W, C)")
    print(f"  target_h={target_h}, max_w={max_w}, new_w={new_w}")

    # 3. For display, resize both to same height just for viewing
    display_h = 300
    oh, ow = img_rgb.shape[:2]
    scale_orig = display_h / oh
    disp_w_orig = int(ow * scale_orig)
    img_orig_disp = cv2.resize(img_bgr, (disp_w_orig, display_h))

    fh, fw = final_rgb.shape[:2]
    scale_final = display_h / fh
    disp_w_final = int(fw * scale_final)
    final_bgr_disp = cv2.resize(cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR),
                                (disp_w_final, display_h),
                                interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Original", img_orig_disp)
    cv2.imshow(f"Final padded (H={target_h}, W={max_w})", final_bgr_disp)
    cv2.waitKey(0)


if __name__ == "__main__":
    # Load from config so you don't desync
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    TARGET_HEIGHT = cfg["data"]["img_height"]
    MAX_WIDTH = cfg["data"]["max_width"]

    TEST_IMAGE_DIR = cfg["data"]["train_dir"]

    image_paths = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.*"))

    if not image_paths:
        print(f"Error: No images found in '{TEST_IMAGE_DIR}'.")
    else:
        print(f"Found {len(image_paths)} images. Visualizing 10 samples.")
        for i, image_path in enumerate(image_paths[:10]):
            visualize_crop(image_path, TARGET_HEIGHT, MAX_WIDTH)
            if i == 9:
                print("Finished 10 samples.")
        cv2.destroyAllWindows()
