import cv2
import os
from src.abhilipsa.stitcher import PanoramaStitcher

def load_images(folder_path):
    """Retrieving images"""
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def generate_panoramas():
    """Generating and saving panoramas for images in specific folders"""
    folders = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
    panorama_creator = PanoramaStitcher()
    
    for folder_name in folders:
        folder_path = f'Images/{folder_name}'
        
        # Loading images and checking their count
        images = load_images(folder_path)
        if len(images) < 2:
            print(f"Folder '{folder_name}' has insufficient images; skipping.")
            continue

        # Creating panorama 
        try:
            panorama, homographies = panorama_creator.make_panorama_for_images_in(images)
            
            # Defining save path and output
            output_path = f'./results/{folder_name}_panorama.jpg'
            os.makedirs('results', exist_ok=True)
            if panorama is not None:
                cv2.imwrite(output_path, panorama)
                print(f"Panorama for '{folder_name}' saved successfully.")
            
            # Displaying homography matrices
            print("\nHomography Information:")
            for i, homography in enumerate(homographies):
                if homography is not None:
                    print(f"From image {i} to {i + 1} homography:\n{homography}")
                else:
                    print(f"Not enough matches for reliable homography between image {i} and {i + 1}")
        
        except Exception as error:
            print(f"Error encountered while creating panorama for '{folder_name}': {error}")

if __name__ == "__main__":
    generate_panoramas()
