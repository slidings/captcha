# src/predict.py
# python -m src.predict path/to/captcha.png
import argparse
import sys
from pathlib import Path

from .model_loader import CaptchaPredictor

def predict_image(image_path):
    """
    CLI function to predict a single CAPTCHA image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        str: Predicted text
    """
    predictor = CaptchaPredictor()
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    prediction = predictor.predict(image_path)
    return prediction

def main():
    parser = argparse.ArgumentParser(description="Predict CAPTCHA text from an image")
    parser.add_argument("image", help="Path to CAPTCHA image")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default="checkpoints_colour/best.pt", 
                       help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    try:
        predictor = CaptchaPredictor(args.config, args.checkpoint)
        prediction = predictor.predict(args.image)
        
        print(f"✅ Prediction: {prediction}")
        print(f"📁 Image: {args.image}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()