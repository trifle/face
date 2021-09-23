"""
Small script to visualize model activations
Uses keract to superimpose activations on existing image
"""
import argparse
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

import keract
from attributes.age_gender import *

parser = argparse.ArgumentParser(description='Visualize activations')
parser.add_argument('--image', type=str, default='image', help='Path to example image')
args = parser.parse_args()

# Assume imdb model
model = imdb_model


def analyze(image_path):
    # Create suitable input image
    image = Image.open(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (64, 64))
    imgs = np.stack([image])
    # generate activations
    activations = keract.get_activations(model, imgs)
    # Set up output directories
    heatmap_directory = Path(image_path).parent / 'heatmap'
    layers_directory = Path(image_path).parent / 'layers'
    heatmap_directory.mkdir(exist_ok=True, parents=True)
    layers_directory.mkdir(exist_ok=True, parents=True)
    # Visualize
    keract.display_activations(activations, save=True, directory=str(layers_directory))
    keract.display_heatmaps(activations, image, save=True, directory=str(heatmap_directory))


if __name__ == '__main__':
    analyze(args.image)
