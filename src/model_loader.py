# src/model_loader.py
import torch
import yaml
import cv2
import numpy as np
import io
from PIL import Image

from .model import CRNN
from .vocab import ITOCH, BLANK
from .decode import greedy_decode
from .transforms import keep_aspect_resize_pad, add_edge_channel, to_float_tensor

class CaptchaPredictor:
    """Main class for CAPTCHA prediction"""
    
    def __init__(self, config_path="config.yaml", checkpoint_path="checkpoints_colour/best.pt"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model Loader] Using device: {self.device}")
        
        # Determine input channels
        self.grayscale = self.cfg["data"]["grayscale"]
        self.input_channels = 1 if self.grayscale else 4  # RGB + edge
        
        # Load model
        self.model = CRNN(
            num_classes=len(ITOCH),
            input_channels=self.input_channels,
            img_height=self.cfg["data"]["img_height"],
            cnn_out=self.cfg["model"]["cnn_out_channels"],
            lstm_hidden=self.cfg["model"]["lstm_hidden"],
            lstm_layers=self.cfg["model"]["lstm_layers"],
            dropout=self.cfg["model"]["dropout"]
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
        print(f"[Model Loader] Model loaded successfully")
        
    def preprocess_image(self, image):
        """
        Preprocess image for inference.
        
        Args:
            image: Can be:
                  - file path (str)
                  - PIL Image
                  - numpy array (H,W,3) RGB
                  - bytes
        
        Returns:
            torch.Tensor: Preprocessed image tensor (1, C, H, W)
        """
        # Convert various input types to numpy array
        if isinstance(image, str):
            # File path
            img_np = cv2.imread(image)
            if img_np is None:
                raise ValueError(f"Could not load image from {image}")
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image, bytes):
            # Bytes
            img_pil = Image.open(io.BytesIO(image))
            img_np = np.array(img_pil.convert('RGB'))
        elif isinstance(image, Image.Image):
            # PIL Image
            img_np = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            # Numpy array
            if len(image.shape) == 2:
                img_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                img_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                img_np = image
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply preprocessing pipeline
        img_resized, _ = keep_aspect_resize_pad(
            img_np, 
            self.cfg["data"]["img_height"], 
            self.cfg["data"]["max_width"]
        )
        
        if not self.grayscale:
            img_resized = add_edge_channel(img_resized)
        
        tensor = to_float_tensor(img_resized)
        tensor = torch.from_numpy(tensor).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict(self, image):
        """
        Run inference on an image.
        
        Args:
            image: Image in any supported format
        
        Returns:
            str: Predicted CAPTCHA text
        """
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                logits, _ = self.model(input_tensor)
                prediction = greedy_decode(logits)[0]  # Get first batch item
            
            return prediction
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, images):
        """
        Run inference on multiple images.
        
        Args:
            images: List of images in any supported format
        
        Returns:
            list: List of predicted strings
        """
        predictions = []
        for img in images:
            try:
                pred = self.predict(img)
                predictions.append(pred)
            except Exception as e:
                predictions.append(f"ERROR: {str(e)}")
        return predictions

# Create a singleton instance
_predictor = None

def get_predictor():
    """Get or create the predictor instance (singleton pattern)"""
    global _predictor
    if _predictor is None:
        _predictor = CaptchaPredictor()
    return _predictor