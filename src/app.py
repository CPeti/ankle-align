"""
Gradio Web Application for Ankle Alignment Classification

A user-friendly web interface for the ankle alignment classification model.
Provides:
- Image upload for single predictions
- Real-time classification results
- Confidence visualization
- Batch processing capability
- REST API endpoint

Usage:
    python src/app.py [--port 7860] [--share]
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import gradio as gr

# Import config - handle both direct run and module import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_DIR,
    IMAGE_SIZE,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    NUM_CLASSES,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    DEVICE,
)
from utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Model Architecture
# ============================================================================

class ResNetTransfer(nn.Module):
    """Transfer learning model using pretrained ResNet18."""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super(ResNetTransfer, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# Model Loading
# ============================================================================

class AnkleClassifier:
    """Wrapper class for the ankle alignment classifier."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.device = DEVICE
        self.model = None
        self.transform = None
        self.label_to_idx = LABEL_TO_IDX
        self.idx_to_label = IDX_TO_LABEL
        
        self._load_model(model_path)
        self._setup_transform()
        
        logger.info(f"Classifier initialized on {self.device}")
    
    def _load_model(self, model_path: Optional[Path] = None):
        """Load the trained model."""
        if model_path is None:
            for path in [MODEL_DIR / "final_model.pth", MODEL_DIR / "best_model.pth"]:
                if path.exists():
                    model_path = path
                    break
        
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                f"Model not found. Please train a model first or provide a valid path."
            )
        
        logger.info(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config from checkpoint
        self.label_to_idx = checkpoint.get('label_to_idx', LABEL_TO_IDX)
        self.idx_to_label = checkpoint.get('idx_to_label', IDX_TO_LABEL)
        
        # Handle string keys
        if self.idx_to_label and isinstance(list(self.idx_to_label.keys())[0], str):
            self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
        
        num_classes = checkpoint.get('num_classes', NUM_CLASSES)
        
        # Create and load model
        self.model = ResNetTransfer(num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _setup_transform(self):
        """Setup preprocessing transform."""
        size = IMAGE_SIZE if isinstance(IMAGE_SIZE, tuple) else (IMAGE_SIZE, IMAGE_SIZE)
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    
    def predict(self, image: Image.Image) -> Tuple[str, dict]:
        """
        Predict ankle alignment from an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (predicted_class, probability_dict)
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        probs = probabilities[0].cpu().numpy()
        
        # Create result dict
        prob_dict = {
            self.idx_to_label[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        # Get predicted class
        predicted_idx = int(np.argmax(probs))
        predicted_class = self.idx_to_label[predicted_idx]
        
        return predicted_class, prob_dict


# ============================================================================
# Gradio Interface
# ============================================================================

# Global classifier instance
classifier: Optional[AnkleClassifier] = None


def load_classifier():
    """Load the classifier (called once at startup)."""
    global classifier
    try:
        classifier = AnkleClassifier()
        return True
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        return False


def classify_image(image: Image.Image) -> Tuple[dict, str]:
    """
    Classify an ankle image.
    
    Args:
        image: Input image from Gradio
        
    Returns:
        Tuple of (label_confidences, result_text)
    """
    if classifier is None:
        return {}, "❌ Model not loaded. Please check the logs."
    
    if image is None:
        return {}, "Please upload an image."
    
    try:
        predicted_class, probabilities = classifier.predict(image)
        confidence = probabilities[predicted_class]
        
        # Create result text with emoji
        emoji = {
            'Pronation': '🔴',
            'Neutral': '🟢', 
            'Supination': '🔵'
        }.get(predicted_class, '⚪')
        
        result_text = f"""
## {emoji} {predicted_class}

**Confidence:** {confidence:.1%}

### What does this mean?

"""
        
        if predicted_class == 'Pronation':
            result_text += """
**Pronation** occurs when the foot rolls inward excessively during walking or running. 
This can lead to:
- Flat feet appearance
- Increased stress on inner foot
- Potential knee and hip alignment issues
"""
        elif predicted_class == 'Neutral':
            result_text += """
**Neutral** alignment is the ideal foot position where the foot rolls inward 
about 15% to absorb shock. This indicates:
- Healthy foot mechanics
- Even weight distribution
- Lower injury risk
"""
        else:  # Supination
            result_text += """
**Supination** (underpronation) occurs when the foot rolls outward. 
This can cause:
- High arches appearance
- Increased stress on outer foot
- Higher impact forces during movement
"""
        
        return probabilities, result_text
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {}, f"❌ Error during classification: {str(e)}"


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    # Custom CSS for styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .result-box {
        padding: 1em;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    footer {
        text-align: center;
        padding: 1em;
        color: #95a5a6;
    }
    """
    
    with gr.Blocks(css=css, title="Ankle Alignment Classifier") as demo:
        
        # Header
        gr.Markdown(
            """
            # 🦶 Ankle Alignment Classifier
            
            Upload an image of feet (posterior view) to classify ankle alignment as 
            **Pronation**, **Neutral**, or **Supination**.
            """
        )
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400,
                )
                
                classify_btn = gr.Button(
                    "🔍 Classify",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    ### Tips for best results:
                    - Use a posterior (back) view of the feet
                    - Ensure good lighting
                    - Keep the ankles clearly visible
                    - Avoid obstructions like pants or shoes
                    """
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Classification Probabilities",
                    num_top_classes=3,
                )
                
                output_text = gr.Markdown(
                    label="Result",
                    value="Upload an image and click **Classify** to see results."
                )
        
        # Examples
        gr.Markdown("### Example Images")
        
        # Check for example images
        example_dir = Path(__file__).parent.parent / "data" / "processed" / "test"
        examples = []
        
        if example_dir.exists():
            for label in ['Pronation', 'Neutral', 'Supination']:
                label_dir = example_dir / label
                if label_dir.exists():
                    images = list(label_dir.glob("*.png"))[:2]
                    examples.extend([[str(img)] for img in images])
        
        if examples:
            gr.Examples(
                examples=examples[:6],
                inputs=input_image,
                outputs=[output_label, output_text],
                fn=classify_image,
                cache_examples=False,
            )
        
        # Event handlers
        classify_btn.click(
            fn=classify_image,
            inputs=input_image,
            outputs=[output_label, output_text],
        )
        
        input_image.change(
            fn=classify_image,
            inputs=input_image,
            outputs=[output_label, output_text],
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            
            <center>
            
            **Ankle Alignment Classifier** | Built with 🤖 PyTorch + 🎨 Gradio
            
            ⚠️ *This tool is for educational purposes only and should not replace professional medical advice.*
            
            </center>
            """
        )
    
    return demo


def create_api_interface() -> gr.Blocks:
    """Create a simple API-focused interface."""
    
    with gr.Blocks(title="Ankle Alignment API") as api_demo:
        gr.Markdown("# Ankle Alignment Classification API")
        
        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil")
            output_json = gr.JSON(label="Prediction Result")
        
        def api_predict(image):
            if image is None:
                return {"error": "No image provided"}
            
            try:
                predicted_class, probabilities = classifier.predict(image)
                return {
                    "predicted_class": predicted_class,
                    "confidence": probabilities[predicted_class],
                    "probabilities": probabilities
                }
            except Exception as e:
                return {"error": str(e)}
        
        input_image.change(fn=api_predict, inputs=input_image, outputs=output_json)
    
    return api_demo


# ============================================================================
# Main
# ============================================================================

def main(
    port: int = 7860,
    share: bool = False,
    api_only: bool = False,
):
    """
    Launch the Gradio application.
    
    Args:
        port: Port to run the server on
        share: Whether to create a public sharing link
        api_only: Whether to use the simple API interface
    """
    
    logger.info("=" * 60)
    logger.info("ANKLE ALIGNMENT WEB APPLICATION")
    logger.info("=" * 60)
    
    # Load classifier
    logger.info("Loading model...")
    if not load_classifier():
        logger.error("Failed to load classifier. Exiting.")
        return
    
    # Create interface
    if api_only:
        demo = create_api_interface()
    else:
        demo = create_interface()
    
    # Launch
    logger.info(f"Starting server on port {port}...")
    logger.info(f"Share link: {'Enabled' if share else 'Disabled'}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Launch Ankle Alignment Classification Web App'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=7860,
        help='Port to run the server on (default: 7860)'
    )
    
    parser.add_argument(
        '--share', '-s',
        action='store_true',
        help='Create a public sharing link'
    )
    
    parser.add_argument(
        '--api-only',
        action='store_true',
        help='Use simple API interface instead of full GUI'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(
        port=args.port,
        share=args.share,
        api_only=args.api_only,
    )

