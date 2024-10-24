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

def main():
    # Load the model
    netG = build_res_unet()
    model = ColorizationModel(netG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint (final_model.pth)
    checkpoint = torch.load("./checkpoints/final_model.pth", map_location=device)
    
    # Check the content of the checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # If the checkpoint is a dict with 'model_state_dict', use that
        state_dict = checkpoint['model_state_dict']
    else:
        # Otherwise, assume the entire checkpoint is the state dict
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # Run the colorization function to create sample outputs
    colorize_random_images(model, VAL_PATH, num_images=15, output_dir='sample_output')

if __name__ == "__main__":
    main()
