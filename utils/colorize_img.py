import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

from config import IMG_SIZE
from utils.convert_to_bw import convert_to_black_and_white
from utils.lab_to_rgb import lab_to_rgb

def colorize_img(model, input_path, output_dir, output_name=None):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        # Check if input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Try to open and process the image
        try:
            original_img = Image.open(input_path)
        except UnidentifiedImageError:
            raise ValueError(f"Unable to open image file: {input_path}. It may be corrupted or in an unsupported format.")

        original_size = original_img.size

        bw_path, bw_img = convert_to_black_and_white(input_path)
        bw_img = bw_img.resize((IMG_SIZE, IMG_SIZE))

        img_tensor = transforms.ToTensor()(bw_img)[:1] * 2. - 1.
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.G(img_tensor)

        colorized_img = lab_to_rgb(img_tensor.cpu(), output.cpu())[0]
        colorized_img = Image.fromarray((colorized_img * 255).astype('uint8'))

        colorized_img = colorized_img.resize(original_size, Image.BICUBIC)

        # Use the provided output_name or default to "results"
        output_name = output_name or "results"
        output_path = os.path.join(output_dir, f"{output_name}.jpg")

        # Ensure the output directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Permission denied: Unable to create output directory {output_dir}")

        # Try to save the image
        try:
            colorized_img.save(output_path, format='JPEG')
        except PermissionError:
            raise PermissionError(f"Permission denied: Unable to save file to {output_path}")

        # Clean up temporary file
        try:
            os.unlink(bw_path)
        except Exception as e:
            print(f"Warning: Unable to delete temporary file {bw_path}. Error: {str(e)}")

        print(f"Colorized image saved as {output_path}")
        return output_path

    except Exception as e:
        print(f"An error occurred during image colorization: {str(e)}")
        return None
