import argparse
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import lab2rgb

from train.pretrain import build_res_unet
from models.models import ColorizationModel
from config import VAL_PATH, IMG_SIZE, PRETRAIN_MODEL_PATH, TRAIN_PATH
from utils.convert_to_bw import convert_to_black_and_white
from utils.lab_to_rgb import lab_to_rgb
import os
import tempfile
from utils.get_test_output import colorize_random_images
from utils.colorize_img import colorize_img

def main():
    parser = argparse.ArgumentParser(description="Colorize a black and white image.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input image")
    parser.add_argument('-o', '--output', default='.', help="Path to the output directory (default: current directory)")
    parser.add_argument('-n', '--name', default="results", help="Name of the output file (without extension)")
    args = parser.parse_args()

    # Load the model
    netG = build_res_unet()
    model = ColorizationModel(netG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load("./checkpoints/final_model.pth", map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # Colorize the input image
    output_path = colorize_img(model, args.input, args.output, args.name)
    print(f"Colorized image saved to: {output_path}")

if __name__ == "__main__":
    main()
