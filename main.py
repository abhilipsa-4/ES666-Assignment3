import cv2
import os
from src.abhilipsa.stitcher import PanaromaStitcher  # Directly importing your specific stitcher

# Define the path to the images
path = 'Images{}*'.format(os.sep)  # Uses OS-compatible path delimiters

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def main():
    # Initialize the PanaromaStitcher from your folder
    stitcher = PanaromaStitcher()
    
    # Loop through each folder in Images
    for impaths in glob.glob(path):
        print(f"\tProcessing {impaths} ...")
        
        # Load images from the current folder
        images = load_images_from_folder(impaths)
        if len(images) < 2:
            print(f"Not enough images in {impaths.split(os.sep)[-1]} to create a panorama. Skipping.")
            continue

        # Create panorama and save the output
        try:
            panorama, homographies = stitcher.make_panaroma_for_images_in(images)
            output_folder = f'./results/{impaths.split(os.sep)[-1]}'
            os.makedirs(output_folder, exist_ok=True)
            
            output_path = f"{output_folder}/panorama.jpg"
            cv2.imwrite(output_path, panorama)
            print(f"Panorama saved for {impaths.split(os.sep)[-1]} at {output_path}")
            print("Homography matrices:", homographies)
        
        except Exception as e:
            print(f"Failed to create panorama for {impaths.split(os.sep)[-1]}. Error: {e}")

if __name__ == "__main__":
    main()
