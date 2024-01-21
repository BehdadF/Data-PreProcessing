import os
import random
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import json
import numpy as np
import math

from datetime import datetime


class Extractor:
    def __init__(self, image_path=None, json_path=None) -> None:
        self.image_path = image_path
        self.json_path = json_path

    def extract_info(self, data):
        '''
        Takes a json file and extracts the labels and coordinates of each segment of an image
        Returns the labels and coordinates
        In oue case, it extracts the coordinate and label of each person's face in a given image.

        Args:
        - data: The json file containing the information about an image

        Returns:
        - labels: A list that contains the labels in an image
        - coords: Coordinates of the rectangle the object exists in an image
        '''
        labels = []
        coords = []
        for person in data['shapes']:
            labels.append(person['label'])
            coords.append((person['points'][0], person['points'][1]))
        return labels, coords
    
    def rotate_image(self, image):
        '''
        Takes an image and rotates it if necessary to ensure correct orientation.
        This function checks Exif data for orientation information and rotates the image accordingly.

        Args:
        - image: PIL Image object.

        Returns:
        - The correctly oriented image.
        '''
        if hasattr(image, '_getexif'):
            exif = image._getexif()

            # If Exif data contains orientation information (tag 274)
            if exif and 274 in exif:
                orientation = exif[274]

                # Rotate/transpose the image based on orientation
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)

        return image

    def normalize_img(self, image_array):
        '''
        Takes a numpy array containing an image and returns the normalized img.

        Args:
        - image_array: Array representation of an image

        Returns:
        - normalized array. 
        '''
        max_value = np.max(image_array)
        return image_array / max_value if max_value > 0 else image_array

    def crop_image(self, image, labels, coords):
        '''
        Takes the original image, labels, and coordinates corresponding to each face in the image.
        Returns a list consisting of tuples in the form of (label, croppedImage).
        The label is a string, and the croppedImage is a numpy array normalized to the range of 0 - 1.
        
        Args:
        - image: PIL Image object.
        - labels: List of strings, each string represents a label.
        - coords: List of tuples, each tuple contains two coordinate tuples ((x1, y1), (x2, y2)).
        
        Returns:
        - img_label: List of tuples (label, croppedImage).
        '''
        if len(labels) != len(coords):
            raise ValueError("Number of labels should match the number of coordinate pairs.")
        
        img_label = []
        for label, coord in zip(labels, coords):
            left = min(coord[0][0], coord[1][0])
            upper = min(coord[0][1], coord[1][1])
            right = max(coord[0][0], coord[1][0])
            lower = max(coord[0][1], coord[1][1])
            
            try:
                cropped = image.crop((left, upper, right, lower))
                cropped_array = np.array(cropped) 
                normalized_array = self.normalize_img(cropped_array) # Normalize to 0-1 range if needed.
                img_label.append((label, normalized_array))
            except Exception as e:
                print(f"Error processing label '{label}': {e}")
        
        return img_label

    def load_json(self, file_path):
        '''
        Takes a path and loads the json file and returns the loaded json file

        Args:
        - file_path: The path that the json file exists. i.e: PATH-TO-JASON/FILENAME.json

        Returns:
        - data loaded from the json file
        '''
        try:
            with open(file_path) as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"JSON file not found at {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON file at {file_path}")
            return None

    # def extract_related_images(self, all_imgs, all_jsons, image_path, json_path):
    def extract_related_images(self, all_imgs, all_jsons):
        '''
        Takes image names, JSON names, and their path and returns a list of tuples 
        containing cropped images and labels, along with the total number of images processed.

        Args:
        all_imgs: A list that contains the names of the images you want to process.
        all_jsons: A list that contains the names of the json files containing the information about the object you want to extract.
            i.e: a file containing labels and coordinates of a face in an image to extract the faces.
        image_path: The path at which the images exist.
        json_path: The path at which the json files exist.
        '''
        all_imgs.sort()
        all_jsons.sort()
        image_label = []
        total_img = 0
        
        for img_name, json_name in zip(all_imgs, all_jsons):
            json_path = os.path.join(self.json_path, json_name)
            img_path = os.path.join(self.image_path, img_name)
            
            json_data = self.load_json(json_path)
            
            if json_data:
                labels, coords = self.extract_info(json_data)
                image = Image.open(img_path)
                image = self.rotate_image(image)
                
                cropped_images = self.crop_image(image, labels, coords)
                total_img += len(cropped_images)
                image_label.append(cropped_images)
        
        return image_label, total_img

    def gather_images(self, image_label):
        '''
        Takes the final list that contains images and extract them into a single list
        '''
        return [image for sublist in image_label for image in sublist]

    def save_images(self, my_images, prefix='./', format='jpg'):
        '''
        Saves a list of images with associated labels to disk.
        Args:
        - my_images (list): A list of tuples containing image-label pairs.
                            Each tuple should have the label as a string and the image as a NumPy array.
        - prefix (str, optional): The path where the images will be saved.
                                Defaults to the current directory './'.
        - format (str, optional): The format for saving images (e.g., 'jpg', 'png').
                                Defaults to 'jpg'.
        Returns:
        - img_paths (list): A list of file paths where the images are saved.
        '''
        img_paths = []
        
        for label, image_data in my_images:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            image_name = f"{timestamp}{random.randint(0, 10**5)}-{label}.{format}"
            path = os.path.join(prefix, image_name)
            
            img = (image_data * 255).astype(np.uint8)
            Image.fromarray(img).save(path)
            
            img_paths.append(path)
        
        return img_paths
    

    def show_images(self, my_images, path='./'):
        '''
        Takes a list of images in numpy format and the path you want to save the result in.

        Args: 
        my_images: A list that contains all images as a numpy array
        '''
        if not my_images:
            print("No images to display.")
            return

        num_images = len(my_images)
        max_cols = 9  # Maximum number of columns
        max_image_size = 4  # Maximum size of each image in inches
        max_figsize = (max_image_size * max_cols, max_image_size * math.ceil(num_images / max_cols))

        num_cols = min(num_images, max_cols)
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=max_figsize)
        axes = axes.flatten()

        for i, (label, image_data) in enumerate(my_images):
            ax = axes[i]
            ax.imshow(image_data)
            ax.axis('off')
            ax.set_title(f'[{i}] Label: {label}')

        for i in range(num_images, num_rows * num_cols):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(path + 'figure.png')
        # plt.savefig('my_plot.png', dpi=300, transparent=True)
        plt.show()
