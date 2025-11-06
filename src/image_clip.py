import cv2
import numpy as np

def remove_horizontal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=5)

    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Create a mask to mark lines for removal
    mask = np.zeros(img.shape[:2], dtype=np.uint8) 

    angle_tolerance = 45 # degrees

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate the angle of the line segment in degrees
            angle_rad = np.arctan2(y2 - y1, x2 - x1) 
            angle_deg = np.degrees(angle_rad) % 180 # Normalize to 0-180 range

            # Check if the line is close to 0 or 180 (horizontal)
            # We need to account for both 0-degree and 180-degree near-horizontal lines
            is_horizontal = (angle_deg < angle_tolerance) or (angle_deg > 180 - angle_tolerance)

            if is_horizontal:
                # Draw the line onto the mask
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3) 

    # Use Inpainting to remove the lines from the original image
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return result

def make_non_white_black(img, white_threshold=250):
    """Convert non-white pixels to black"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Create binary mask: white pixels = 255, non-white = 0
    _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    
    # Convert back to 3-channel if original was 3-channel
    if len(img.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return binary

def get_lm_rm(img):
    """Get leftmost and rightmost coordinates using image processing (no YOLO)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    content_mask = gray < 255
    
    if not np.any(content_mask):
        return None, None
    
    content_cols = np.any(content_mask, axis=0)
    
    if not np.any(content_cols):
        return None, None
    
    leftmost = float(np.argmax(content_cols))
    rightmost = float(len(content_cols) - 1 - np.argmax(content_cols[::-1]))  # Last column with content
    
    return leftmost, rightmost

def clip_image_to_text(original_img, leftmost, rightmost, padding=10):
    """Clip original image to text boundaries with optional padding"""
    
    # Get image dimensions
    height, width = original_img.shape[:2]
    
    # Add padding and ensure within image bounds
    left_clip = max(0, int(leftmost - padding))
    right_clip = min(width, int(rightmost + padding))
    
    # Clip the image (keep full height, clip width)
    clipped_img = original_img[:, left_clip:right_clip]
    
    return clipped_img, (left_clip, right_clip)

def clip_image_normal(img):
    img = make_non_white_black(img, 250)
    img = remove_horizontal(img)
    lm, rm = get_lm_rm(img)
    if lm == None and rm == None:
        return img
    else:
        clipped_img, _  = clip_image_to_text(img, lm, rm)
        return clipped_img
    
def resize_with_padding(img,  target_height, target_width, padding_color=255):
    """
    Resize image to target dimensions while preserving aspect ratio.
    Add padding to fill the remaining space.
    
    Args:
        img: Input image
        target_width: Desired width
        target_height: Desired height
        padding_color: Color for padding (255=white, 0=black, [B,G,R] for color)
    """
    h, w = img.shape[:2]
    
    # Calculate scale to fit within target dimensions
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target dimensions
    if len(img.shape) == 3:  # Color image
        if isinstance(padding_color, int):
            canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
        else:
            canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
    else:  # Grayscale
        canvas = np.full((target_height, target_width), padding_color, dtype=np.uint8)
    
    # Calculate position to center the resized image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return canvas, (x_offset, y_offset, new_width, new_height)