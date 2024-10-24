import os
import tempfile
from PIL import Image

def convert_to_black_and_white(image_path):
    with Image.open(image_path) as img:
        bw_img = img.convert('L')
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        bw_img.save(temp_path)
    return temp_path, bw_img

