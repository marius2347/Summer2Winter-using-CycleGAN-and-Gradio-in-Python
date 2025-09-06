# import necessary libraries
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# block definition for the residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# generator definition
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super().__init__()

        # initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        ## downsampling
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels * 2

        # residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]

        # upsampling
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels // 2

        # output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# image transformation class
class ImageTransformer:
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = ResnetGenerator().to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.netG.load_state_dict(state_dict, strict=False)
        self.netG.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

# method to transform image
    def transform_image(self, img):
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.netG(x)
        y = (y.squeeze(0).cpu() + 1) / 2 
        y = transforms.ToPILImage()(y)
        return y

# functions to apply specific filters
def summer_filter(img, model_path="./checkpoints/winter2summer_yosemite_pretrained/latest_net_G.pth"):
    transformer = ImageTransformer(model_path)
    return transformer.transform_image(img)

def winter_filter(img, model_path="./checkpoints/summer2winter_yosemite_pretrained/latest_net_G.pth"):
    transformer = ImageTransformer(model_path)
    return transformer.transform_image(img)
