import cv2
import os
from src.abhilipsa.stitcher import PanaromaStitcher  

def load_images(image_folder):
    # Load images from the specified folder
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_folder, filename))
            if img is not None:
                images.append(img)
    return images

def save_results(stitched_image, homographies, output_folder="results"):
    # Create output folder if it does not exist
    os.makedirs('./results', exist_ok=True)

    # Save the stitched image
    cv2.imwrite(os.path.join(output_folder, "stitched_image.jpg"), stitched_image)

    # Save homography matrices as text files
    for i, H in enumerate(homographies):
        filename = os.path.join(output_folder, f"homography_{i+1}.txt")
        with open(filename, 'w') as f:
            for row in H:
                f.write(' '.join(map(str, row)) + '\n')

def main():
    # Load images from the "Images" folder
    image_folder = "Images"
    images = load_images(image_folder)
    
    if len(images) < 2:
        print("Need at least two images for stitching.")
        return

    # Initialize the PanoramaStitcher
    stitcher = PanaromaStitcher()
    
    try:
        # Stitch images and obtain homographies
        stitched_image, homographies = stitcher.make_panaroma_for_images_in(images)

        # Save the results
        save_results(stitched_image, homographies)

        print("Panorama stitching complete. Check the 'results' folder for output.")
    
    except Exception as e:
        print("An error occurred during stitching:", e)

if __name__ == "__main__":
    main()
