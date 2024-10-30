import cv2
import os
from src.abhilipsa.stitcher import PanoramaStitcher  # Corrected typo

<<<<<<< HEAD
# Initialize the PanoramaStitcher
stitcher = PanoramaStitcher()

# Directory containing the images folder
images_folder = "Images"

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Create panoramas for folders I1 to I6
for i in range(1, 7):
    # Construct folder path as a string
    folder_path = os.path.join(images_folder, f"I{i}")
    
    # Check if folder_path exists to avoid errors
    if not os.path.isdir(folder_path):
        print(f"Directory {folder_path} does not exist.")
        continue

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
        print(f"Error encountered while creating panorama for '{folder_path}': {e}")
=======
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
>>>>>>> parent of 935afac (major changes-2)
