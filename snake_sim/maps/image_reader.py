from PIL import Image
import numpy as np
from pathlib import Path


def read_image_map(map_path)
    map_name = 'map1.png'
    map_dir = Path(__file__).parent / 'map_images'
    image = Image.open(map_dir / map_name)
    image_matrix = np.array(image)

# Print the shape of the matrix
print(f'Shape of the matrix: {image_matrix.shape}')

# Display a portion of the matrix
print('Matrix values:')
print(image_matrix)