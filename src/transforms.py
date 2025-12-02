import cv2
import numpy as np
import glob
import os


def keep_aspect_resize_pad(img, target_h=32, max_w=128):
    """
    Improved version:
    - Removes black noise lines
    - Boosts contrast for pale coloured characters using LAB + CLAHE
    - Finds tight bounding box around characters
    - Resizes with aspect ratio preserved
    - Pads to (target_h, max_w)
    
    img: RGB image (as in your dataset pipeline)
    """

    # -----------------------------
    # 1. Remove black noise lines
    # -----------------------------
    # Black line is (0,0,0)
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(ab_combo)

    # -----------------------------
    # 3. Adaptive threshold → clean mask for cropping
    # -----------------------------
    mask = cv2.adaptiveThreshold(
        enh, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 3
    )

    # -----------------------------
    # 4. Find tight bounding box
    # -----------------------------
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # fallback: no content detected
        img_crop = img_clean
    else:
        top, bottom = ys.min(), ys.max()
        left, right = xs.min(), xs.max()
        img_crop = img_clean[top:bottom+1, left:right+1]

    # -----------------------------
    # 5. Resize with aspect ratio
    # -----------------------------
    h, w = img_crop.shape[:2]

    scale = min(target_h / h, max_w / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # -----------------------------
    # 6. Pad to target size
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
    Convert image to float tensor-style numpy array (CHW, normalized).
    """
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img - mean) / std
    return np.transpose(img, (2, 0, 1))

# --- UPDATED FUNCTION TO VISUALIZE THE CROP ---
def visualize_crop(image_path, target_h, max_w):
    """
    Loads an image, runs the cropping logic from keep_aspect_resize_pad,
    and displays the original image with the crop-box and the final 
    resized/padded image.
    """
    
    # --- 1. Load Image ---
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    img_with_box = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- 2. Run Cropping Logic (Updated) ---
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Use Otsu's method to automatically find the best threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- UPDATED BOUNDING BOX LOGIC ---
    # Find all contours (shapes) in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the bounding box of *all* contours combined
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        # Draw the single, combined bounding box
        cv2.rectangle(img_with_box, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    else:
        print("No content found to crop.")
    # --- END UPDATE ---

    # --- 3. Run Resize/Pad Logic ---
    final_padded_img, _ = keep_aspect_resize_pad(img_rgb, target_h=target_h, max_w=max_w)
    
    # --- 4. Display Results ---
    display_h = 300
    orig_h, orig_w = img_with_box.shape[:2]
    scale = display_h / orig_h
    display_w = int(orig_w * scale)
    
    img_with_box_resized = cv2.resize(img_with_box, (display_w, display_h))

    # --- Visualization Updated ---
    # The final image is now (target_h, max_w). 
    # We can resize it for better viewing.
    # Let's make its display width match the original's display width.
    final_display_h = int(target_h * (display_w / max_w))
    
    final_padded_img_display = cv2.resize(final_padded_img, (display_w, final_display_h), 
                                          interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Original with Bounding Box", img_with_box_resized)
    cv2.imshow(f"Final Padded Output (H={target_h}, W={max_w})", final_padded_img_display)
    
    print(f"Showing results for: {os.path.basename(image_path)}")
    print(f"Target H: {target_h}, Max W: {max_w}")
    print("Press any key to close and test the next image...")
    cv2.waitKey(0)


if __name__ == "__main__":
    # This block will run when you execute `python src/transforms.py`
    
    # --- CONFIGURATION ---
    TEST_IMAGE_DIR = "data/train" 
    TARGET_HEIGHT = 32
    MAX_WIDTH = 192     
    # --- END CONFIGURATION ---

    image_paths = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.*"))
    
    if not image_paths:
        print(f"Error: No images found in '{TEST_IMAGE_DIR}'.")
        print("Please update the TEST_IMAGE_DIR variable in this script.")
    else:
        print(f"Found {len(image_paths)} images. Visualizing 10 samples.")
        
        for i, image_path in enumerate(image_paths[:10]):
            visualize_crop(image_path, TARGET_HEIGHT, MAX_WIDTH)
            if i == 9:
                print("Finished 10 samples.")

        cv2.destroyAllWindows()
