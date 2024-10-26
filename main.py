import cv2
import os
from src.abhilipsa.stitcher import PanoramaStitcher  # Corrected typo

def load_images(folder_path):
    img_list = []
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            img_list.append(image)
    return img_list

def main():
    # Specify the folders containing images
    image_folders = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
    
    # Instantiate the panorama stitcher
    stitcher = PanoramaStitcher()
    
    # Iterate through each image folder
    for folder in image_folders:
        path = f'Images/{folder}'
        
        # Load images from the folder
        images = load_images(path)
        if len(images) < 2:
            print(f"Not enough images in folder {folder} to create a panorama. Skipping.")
            continue
        
        # Attempt to create the panorama
        try:
            panorama_image, homographies = stitcher.create_panorama(images)
            
            # Save the stitched image
            result_path = f'./results/{folder}_panorama.jpg'
            if not os.path.exists('results'):
                os.makedirs('results')
            cv2.imwrite(result_path, panorama_image)
            print(f"Panorama for {folder} has been successfully created and saved!")

            # Display the calculated homography matrices
            print("Homography Matrices:")
            for i, H in enumerate(homographies):
                if H is not None:
                    print(f"Homography between image {i} and {i + 1}:")
                    print(H)
                else:
                    print(f"Homography between image {i} and {i + 1}: Not computed (not enough matches).")
        
        except Exception as e:
            print(f"Error while creating panorama for folder {folder}: {e}")

if __name__ == "__main__":
    main()
