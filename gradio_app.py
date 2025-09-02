# app.py
import gradio as gr
import os
import requests
import threading
import time
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2

# Import our CycleGAN model (will work with pre-trained weights)
try:
    from cyclegan_model import CycleGAN, ImageTransformer, train_cyclegan_model
    CYCLEGAN_AVAILABLE = True
except:
    CYCLEGAN_AVAILABLE = False
    print("CycleGAN not available, using alternative methods")

class WebApp:
    def __init__(self):
        self.transformer = ImageTransformer() if CYCLEGAN_AVAILABLE else None
        self.training_status = "Not started"
        self.is_training = False
        self.download_pretrained_models()

    def download_pretrained_models(self):
        """Download pre-trained models if available"""
        model_urls = {
            "summer2winter": "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/releases/download/v0.1/latest_net_G_A.pth",
            # Add more pre-trained models as needed
        }
        
        os.makedirs("pretrained_models", exist_ok=True)
        
        # For now, we'll use alternative methods if no pre-trained model
        print("Pre-trained models will be downloaded when available")

    def simple_winter_transform(self, image):
        """Simple winter transformation using image processing (fallback)"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Apply winter effects
        # 1. Reduce saturation (make it more gray/blue)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3  # Reduce saturation
        img_desaturated = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # 2. Add blue tint
        img_blue_tint = img_desaturated.copy().astype(np.float32)
        img_blue_tint[:, :, 0] *= 0.9  # Reduce red
        img_blue_tint[:, :, 1] *= 0.95  # Slightly reduce green
        img_blue_tint[:, :, 2] *= 1.1  # Increase blue
        img_blue_tint = np.clip(img_blue_tint, 0, 255).astype(np.uint8)
        
        # 3. Increase brightness slightly (snow effect)
        enhancer = ImageEnhance.Brightness(Image.fromarray(img_blue_tint))
        img_bright = enhancer.enhance(1.2)
        
        # 4. Add slight blur for atmospheric effect
        img_final = img_bright.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img_final

    def simple_summer_transform(self, image):
        """Simple summer transformation using image processing (fallback)"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Apply summer effects
        # 1. Increase saturation
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
        img_saturated = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # 2. Add warm tint
        img_warm = img_saturated.copy().astype(np.float32)
        img_warm[:, :, 0] *= 1.1  # Increase red
        img_warm[:, :, 1] *= 1.05  # Slightly increase green
        img_warm[:, :, 2] *= 0.9  # Reduce blue
        img_warm = np.clip(img_warm, 0, 255).astype(np.uint8)
        
        # 3. Increase contrast
        enhancer = ImageEnhance.Contrast(Image.fromarray(img_warm))
        img_contrast = enhancer.enhance(1.2)
        
        # 4. Sharpen slightly
        img_final = img_contrast.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        return img_final
        """Handle image transformation with progress"""
        if image is None:
            return None, "❌ Please upload an image first!"
        
        progress(0.1, desc="Loading model...")
        
        # Try to load model
        if not self.transformer.load_model():
            return None, "❌ Model not found! Please train the model first or download pre-trained weights."
        
        progress(0.5, desc="Transforming image...")
        
        try:
            result_image, status = self.transformer.transform_image(image, direction)
            progress(1.0, desc="Complete!")
            
            if result_image:
                return result_image, f"✅ {status}"
            else:
                return None, f"❌ {status}"
                
        except Exception as e:
            return None, f"❌ Transformation error: {str(e)}"

    def start_training_handler(self, summer_path, winter_path, epochs, batch_size, max_dataset_size):
        """Handle training in a separate thread"""
        if self.is_training:
            return "⚠️ Training already in progress!"

        def train_thread():
            try:
                self.is_training = True
                self.training_status = "🚀 Starting training..."
                
                # Validate inputs
                if not summer_path or not winter_path:
                    self.training_status = "❌ Please provide both summer and winter image paths"
                    self.is_training = False
                    return

                if not os.path.exists(summer_path):
                    self.training_status = f"❌ Summer path not found: {summer_path}"
                    self.is_training = False
                    return

                if not os.path.exists(winter_path):
                    self.training_status = f"❌ Winter path not found: {winter_path}"
                    self.is_training = False
                    return

                self.training_status = "📊 Preparing dataset..."
                time.sleep(1)

                # Start training
                self.training_status = "🏋️ Training in progress... This may take several hours."
                
                model = train_cyclegan_model(
                    summer_path=summer_path,
                    winter_path=winter_path,
                    num_epochs=int(epochs),
                    batch_size=int(batch_size),
                    max_dataset_size=int(max_dataset_size) if max_dataset_size else None
                )
                
                self.training_status = "✅ Training completed successfully! Model saved to checkpoints/final"
                
            except Exception as e:
                self.training_status = f"❌ Training failed: {str(e)}"
            finally:
                self.is_training = False

        # Start training in background thread
        training_thread = threading.Thread(target=train_thread)
        training_thread.daemon = True
        training_thread.start()
        
        return "🚀 Training started! Check status below..."

    def get_training_status(self):
        """Get current training status"""
        return self.training_status

    def create_interface(self):
        """Create the main Gradio interface"""
        
        # Custom CSS for beautiful styling
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', system-ui, sans-serif;
            max-width: 1200px !important;
            margin: 0 auto;
        }
        
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .feature-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        
        .demo-section {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .training-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 15px;
            border: 2px dashed #667eea;
            margin: 1rem 0;
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 2rem;
            background: #2c3e50;
            color: white;
            border-radius: 10px;
        }
        """

        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Summer↔Winter CycleGAN") as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🌞❄️ Summer ↔ Winter CycleGAN Transformer</h1>
                <p style="font-size: 1.2em; margin: 1rem 0;">
                    Transform your summer landscapes into winter wonderlands using deep learning!
                </p>
                <p style="opacity: 0.9;">
                    Powered by PyTorch CycleGAN | Built with ❤️ for seasonal magic
                </p>
            </div>
            """)

            # Main demo section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="demo-section">')
                    
                    gr.HTML("""
                    <div class="feature-box">
                        <h3>📤 Upload Your Image</h3>
                        <p>Upload a landscape image to transform it between seasons!</p>
                        <p><strong>💡 No GPU? No problem!</strong> Use image processing mode for instant results.</p>
                    </div>
                    """)
                    
                    input_image = gr.Image(
                        type="pil",
                        label="🖼️ Input Image",
                        height=350,
                        sources=["upload", "webcam"],
                        interactive=True
                    
                    use_ai_toggle = gr.Checkbox(
                        label="🤖 Use AI Model (if available)",
                        value=True,
                        info="Uncheck to use fast image processing instead"
                    )
                    
                    direction = gr.Radio(
                        choices=[
                            ("🌞 Summer → ❄️ Winter", "summer_to_winter"),
                            ("❄️ Winter → 🌞 Summer", "winter_to_summer")
                        ],
                        value="summer_to_winter",
                        label="🔄 Transformation Direction",
                        interactive=True
                    )
                    
                    transform_btn = gr.Button(
                        "🎨 Transform Image",
                        variant="primary",
                        size="lg",
                        scale=1
                    )
                    
                    gr.HTML('</div>')

                with gr.Column(scale=1):
                    gr.HTML('<div class="demo-section">')
                    
                    gr.HTML("""
                    <div class="feature-box">
                        <h3>🎯 Transformed Result</h3>
                        <p>Your AI-generated seasonal transformation</p>
                    </div>
                    """)
                    
                    output_image = gr.Image(
                        label="✨ Transformed Image",
                        height=350,
                        interactive=False
                    )
                    
                    status_output = gr.Textbox(
                        label="📊 Status",
                        value="Ready for transformation! Upload an image to begin.",
                        interactive=False,
                        lines=2
                    )
                    
                    download_btn = gr.DownloadButton(
                        "💾 Download Result",
                        variant="secondary",
                        visible=False
                    )
                    
                    gr.HTML('</div>')

            # Examples section
            gr.HTML("""
            <div