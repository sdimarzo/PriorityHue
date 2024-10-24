import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

from config import IMG_SIZE
from utils.convert_to_bw import convert_to_black_and_white
from utils.lab_to_rgb import lab_to_rgb

def colorize_random_images(model, val_path, num_images=72, output_dir='sample_output'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from the validation path
    image_files = [f for f in os.listdir(val_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly select 'num_images' from the list
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    for idx, img_name in enumerate(tqdm(selected_images, desc="Colorizing images")):
        img_path = os.path.join(val_path, img_name)
        
        # Convert to black and white
        bw_path, bw_img = convert_to_black_and_white(img_path)
        bw_img = bw_img.resize((IMG_SIZE, IMG_SIZE))
        
        # Prepare input for the model
        img_tensor = transforms.ToTensor()(bw_img)[:1] * 2. - 1.
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Generate colorized output
        with torch.no_grad():
            output = model.G(img_tensor)
        
        # Convert output to RGB
        colorized_img = lab_to_rgb(img_tensor.cpu(), output.cpu())[0]
        colorized_img = Image.fromarray((colorized_img * 255).astype(np.uint8))
        
        # Create a new image with both BW and colorized versions side by side
        combined_img = Image.new('RGB', (IMG_SIZE * 2, IMG_SIZE))
        combined_img.paste(bw_img, (0, 0))
        combined_img.paste(colorized_img, (IMG_SIZE, 0))
        
        # Save the combined image
        output_filename = f"sample_{idx+1:03d}.png"
        combined_img.save(os.path.join(output_dir, output_filename))
        
        # Clean up temporary file
        os.unlink(bw_path)
    
    print(f"Saved {len(selected_images)} sample images in {output_dir}")

# You can add more utility functions related to test output here if needed
