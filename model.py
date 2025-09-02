import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ImageTransformer:
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.load_pytorch_hub_model()

    def load_pytorch_hub_model(self):
        try:
            print("Loading CycleGAN from PyTorch Hub...")
            self.model = torch.hub.load('pytorch/vision', 'cyclegan_summer2winter', pretrained=True)
            self.model.to(device)
            self.model.eval()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"PyTorch Hub model failed: {e}")
            try:
                print("Trying alternative model...")
                self.model = torch.hub.load('nicolalandro/cyclegan_pytorch', 'cyclegan', pretrained=True, map_location=device)
                self.model.eval()
                print("Alternative model loaded!")
                return True
            except Exception as e2:
                print(f"Alternative model failed: {e2}")
                return False

    def transform(self, image, direction="summer_to_winter"):
        if self.model is None:
            if not self.load_pytorch_hub_model():
                return None, "Model not available"

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            if hasattr(self.model, 'G_AB') and hasattr(self.model, 'G_BA'):
                if direction == "summer_to_winter":
                    output_tensor = self.model.G_AB(input_tensor)
                else:
                    output_tensor = self.model.G_BA(input_tensor)
            else:
                output_tensor = self.model(input_tensor)

        output_tensor = (output_tensor + 1) / 2.0
        output_tensor = torch.clamp(output_tensor, 0, 1)
        
        output_image = output_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_image = (output_image * 255).astype(np.uint8)
        
        return Image.fromarray(output_image), "AI transformation successful"

def winter_filter(image):
    img_array = np.array(image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.2, 0, 255)
    
    img_result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    blue_tint = img_result.astype(np.float32)
    blue_tint[:, :, 0] *= 0.9
    blue_tint[:, :, 1] *= 0.95
    blue_tint[:, :, 2] *= 1.1
    blue_tint = np.clip(blue_tint, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blue_tint)

def summer_filter(image):
    img_array = np.array(image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.4, 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.1, 0, 255)
    
    img_result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    warm_tint = img_result.astype(np.float32)
    warm_tint[:, :, 0] *= 1.1
    warm_tint[:, :, 1] *= 1.05
    warm_tint[:, :, 2] *= 0.9
    warm_tint = np.clip(warm_tint, 0, 255).astype(np.uint8)
    
    return Image.fromarray(warm_tint)