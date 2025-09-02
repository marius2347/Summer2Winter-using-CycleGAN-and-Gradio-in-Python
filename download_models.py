# Download the models for Summer2Winter Project
# download_models.py
import os
import requests
import gdown
from pathlib import Path
from tqdm import tqdm
import zipfile

class ModelDownloader:
    """Download pre-trained CycleGAN models"""
    
    def __init__(self):
        self.models_dir = Path("checkpoints")
        self.models_dir.mkdir(exist_ok=True)
        
        # Available pre-trained models (real working links)
        self.available_models = {
            "summer2winter_yosemite": {
                "description": "Summer to Winter Yosemite (Official PyTorch-CycleGAN)",
                "url": "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/releases/download/v0.1/summer2winter_yosemite.zip",
                "size": "180MB",
                "type": "zip"
            },
            "winter2summer_yosemite": {
                "description": "Winter to Summer Yosemite (Official PyTorch-CycleGAN)",
                "url": "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/releases/download/v0.1/winter2summer_yosemite.zip", 
                "size": "180MB",
                "type": "zip"
            },
            "horse2zebra": {
                "description": "Horse to Zebra (for testing)",
                "url": "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/releases/download/v0.1/horse2zebra.zip",
                "size": "180MB", 
                "type": "zip"
            }
        }

    def download_file(self, url, filename):
        """Download file with progress bar"""
        try:
            print(f"📥 Downloading from: {url}")
            
            if "drive.google.com" in url:
                # Use gdown for Google Drive
                gdown.download(url, str(filename), quiet=False)
            else:
                # Regular download with progress
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filename, 'wb') as file, tqdm(
                    desc=os.path.basename(filename),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            progress_bar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False

    def extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        try:
            print(f"📦 Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            # Remove zip file after extraction
            os.remove(zip_path)
            print(f"✅ Extracted and cleaned up")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting {zip_path}: {e}")
            return False

    def convert_pytorch_models(self, model_dir):
        """Convert official PyTorch models to our format"""
        try:
            # Find the model files
            model_files = list(Path(model_dir).rglob("*.pth"))
            
            if not model_files:
                print("❌ No .pth files found in extracted directory")
                return False
            
            # Create our checkpoints structure
            final_dir = self.models_dir / "final"
            final_dir.mkdir(exist_ok=True)
            
            # Copy and rename files to our convention
            for model_file in model_files:
                if "netG_A" in model_file.name:  # Summer to Winter generator
                    target = final_dir / "G_SW.pth"
                    target.write_bytes(model_file.read_bytes())
                    print(f"✅ Copied {model_file.name} -> G_SW.pth")
                    
                elif "netG_B" in model_file.name:  # Winter to Summer generator  
                    target = final_dir / "G_WS.pth"
                    target.write_bytes(model_file.read_bytes())
                    print(f"✅ Copied {model_file.name} -> G_WS.pth")
            
            return True
            
        except Exception as e:
            print(f"❌ Error converting models: {e}")
            return False

    def download_model(self, model_name):
        """Download a specific model"""
        if model_name not in self.available_models:
            print(f"❌ Model '{model_name}' not available")
            return False
        
        model_info = self.available_models[model_name]
        print(f"\n📦 Downloading: {model_info['description']}")
        print(f"📊 Size: {model_info['size']}")
        
        # Download file
        filename = self.models_dir / f"{model_name}.zip"
        success = self.download_file(model_info["url"], filename)
        
        if not success:
            return False
        
        # Extract if it's a zip
        if model_info["type"] == "zip":
            extract_dir = self.models_dir / f"{model_name}_extracted"
            if self.extract_zip(filename, extract_dir):
                # Convert to our format
                self.convert_pytorch_models(extract_dir)
        
        print(f"✅ {model_name} setup completed!")
        return True

    def list_available_models(self):
        """List all available pre-trained models"""
        print("\n🤖 Available Pre-trained Models:")
        print("=" * 60)
        
        for name, info in self.available_models.items():
            print(f"📦 {name}")
            print(f"   📝 Description: {info['description']}")
            print(f"   📊 Size: {info['size']}")
            
            # Check if downloaded
            final_model = self.models_dir / "final"
            if final_model.exists() and (final_model / "G_SW.pth").exists():
                print("   ✅ Status: Downloaded and ready!")
            else:
                print("   ❌ Status: Not downloaded")
            print()

    def check_model_exists(self):
        """Check if we have a working model"""
        final_model = self.models_dir / "final"
        g_sw = final_model / "G_SW.pth"
        g_ws = final_model / "G_WS.pth"
        
        return g_sw.exists() and g_ws.exists()

    def quick_setup(self):
        """Quick setup - download the best model for demo"""
        print("🚀 Quick Setup: Downloading summer2winter_yosemite model...")
        
        if self.check_model_exists():
            print("✅ Model already exists! Ready to use.")
            return True
        
        success = self.download_model("summer2winter_yosemite")
        
        if success:
            print("🎉 Quick setup completed! You can now use the app.")
        else:
            print("❌ Quick setup failed. You can still use image processing mode.")
            
        return success

def main():
    """Main function for command line usage"""
    downloader = ModelDownloader()
    
    print("🌞❄️ CycleGAN Model Downloader")
    print("=" * 50)
    
    # Check current status
    if downloader.check_model_exists():
        print("✅ CycleGAN models are already installed!")
        print("🚀 You can run: python app.py")
        return
    
    # Show available models
    downloader.list_available_models()
    
    print("\n💡 Options:")
    print("1. Type 'quick' for automatic setup (recommended)")
    print("2. Type model name to download specific model")
    print("3. Type 'all' to download all models")
    print("4. Press Enter to skip and use image processing only")
    
    choice = input("\n👉 Your choice: ").strip().lower()
    
    if choice == "quick":
        downloader.quick_setup()
    elif choice == "all":
        for model_name in downloader.available_models:
            downloader.download_model(model_name)
    elif choice in downloader.available_models:
        downloader.download_model(choice)
    elif choice == "":
        print("⏭️ Skipping model download. You can use image processing mode in the app.")
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    print("\n🎉 Setup completed!")
    print("🚀 Run the app with: python app.py")

if __name__ == "__main__":
    main()