import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import torchvision.models
from torch.autograd import Variable
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.res_scale = 0.2
        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, 64, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.ModuleList([
            nn.ModuleList([
                block(in_features=(i + 1) * 64, non_linearity=(i != 4))
                for i in range(5)
            ]) for _ in range(23 * 3)
        ])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        upsample_layers = [
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(),
        ]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        output1 = self.conv1(x)
        output = output1
        for idx, dense_block_group in enumerate(self.res_blocks):
            inputs = output
            for block in dense_block_group:
                out = block(inputs)
                inputs = torch.cat([inputs, out], 1)
            output = out.mul(self.res_scale) + output
            if (idx + 1) % 3 == 0:  # apply residual connection every 3 groups
                output = output.mul(self.res_scale) + output1
                output1 = output
        output2 = self.conv2(output)
        output = torch.add(output1, output2)
        output = self.upsampling(output)
        output = self.conv3(output)
        return output
def denormalize(tensors,mean,std):
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)
def validate_input(input_image_path, output_dir, model_path):
    assert os.path.exists(input_image_path), f"Image not found at {input_image_path}"
    assert os.path.getsize(input_image_path) < 500*1024, "Image size must be less than 500KB"
    img = Image.open(input_image_path)
    assert img.mode == 'RGB', "Image must be in RGB format"
    assert img.size[0] <= 1000 and img.size[1] <= 1000, "Image size must be less than 1000x1000"
    assert os.path.exists(output_dir), f"Output directory does not exist: {output_dir}"
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert model_path.endswith('.pth'), "Model file must be a '.pth' file"
def jcf_esrganx4(input_image_path,output_image_path,esrgan_model_path):
    try:
        output_dir = os.path.dirname(output_image_path)
        validate_input(input_image_path, output_dir, esrgan_model_path)
        # The rest of your code here...

        opt_image_path=True 
        opt_checkpoint_model=True 
        opt_channels=3 
        opt_residual_blocks=23
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_image_name= os.path.basename(input_image_path)
        output_image_name=os.path.basename(output_image_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        esrgan_generator = Generator().to(device)
        esrgan_generator.load_state_dict(torch.load(esrgan_model_path, map_location=device))
        esrgan_generator.eval()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        input_img = Image.open(input_image_path).convert('RGB')
        image_tensor = Variable(transform(input_img)).to(device).unsqueeze(0)
        with torch.no_grad():
            sr_img_esrgan = denormalize(esrgan_generator(image_tensor),mean,std).cpu().squeeze(0).detach()
        sr_img_esrgan = Image.fromarray(sr_img_esrgan.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy())
        sr_img_esrgan.save(output_image_path)
        print("---------------------------------------------------------")
        print("Input:",input_image_path,": Size =",input_img.size)
        print("Output:",output_image_path,": Size =",sr_img_esrgan.size)
        print("---------------------------------------------------------")
    except AssertionError as error:
        print(f"Validation Error: {error}")
    except IOError as error:
        print(f"I/O error({error.errno}): {error.strerror}")
    except Exception as error:
        print(f"Unexpected error: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESRGAN model for image enhancement')
    parser.add_argument('-i', '--input_image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('-o', '--output_image_path', type=str, required=True, help='Path to save the output image')
    parser.add_argument('-model', '--esrgan_model_path', type=str, required=True, help='Path to ESRGAN model')
    args = parser.parse_args()
    jcf_esrganx4(args.input_image_path,args.output_image_path,args.esrgan_model_path)
