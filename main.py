import cv2
import os
from src.abhilipsa.stitcher import PanoramaStitcher

# Initialize the PanoramaStitcher
stitcher = PanoramaStitcher()

# Path to the images folder
images_folder = "Images"

# Create output folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Create panorama for each folder I1 to I6
for i in range(1, 7):
    folder_path = os.path.join(images_folder, f"I{i}")
    try:
        stitched_image, homographies = stitcher.make_panorama_for_images_in(folder_path)

        # Save the stitched image
        output_path = os.path.join("results", f"panorama_{i}.jpg")
        cv2.imwrite(output_path, stitched_image)
        print(f"Panorama saved for {folder_path} at {output_path}")

        # Print homography matrices
        for idx, H in enumerate(homographies):
            print(f"Homography matrix {idx+1} for {folder_path}:\n", H)

    except Exception as e:
        print(f"Could not create panorama for {folder_path}. Error: {e}")