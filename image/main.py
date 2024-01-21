from image.Extractor import Extractor  # Assuming your class is defined in a file named Extractor.py
import os

def main():
    # Prompt user for input paths and output path
    image_path = input("Enter the path where the images are located: ")
    json_path = input("Enter the path where the JSON files are located: ")
    output_path = input("Enter the path where you want to save the output images: ")

    # List all image and JSON files in the input paths
    all_imgs = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.jpeg')]
    all_jsons = [f for f in os.listdir(json_path) if f.endswith('.json')]

    # Create an instance of the Extractor class
    extractor = Extractor(image_path=image_path, json_path=json_path)

    # Extract and process images
    image_label, total_images = extractor.extract_related_images(all_imgs, all_jsons)

    # Gather images into a single list
    my_images = extractor.gather_images(image_label)

    extractor.save_images(my_images, output_path, format="jpg")

    print(f"Total {total_images} images processed and saved to {output_path}")

if __name__ == "__main__":
    main()
