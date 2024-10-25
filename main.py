import cv2
import os
from src.abhilipsa.stitcher import PanaromaStitcher

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def main():
    # Initialize the stitcher
    stitcher = PanaromaStitcher()

    # Load images directly from the main Images folder
    images = load_images_from_folder('Images')
    if len(images) < 2:
        print("Not enough images to create a panorama.")
        return

    # Create panorama
    try:
        panorama, homographies = stitcher.make_panaroma_for_images_in(images)
        output_path = './results/panorama.jpg'
        os.makedirs('results', exist_ok=True)
        cv2.imwrite(output_path, panorama)
        print("Panorama created and saved!")
    except Exception as e:
        print(f"Failed to create panorama. Error: {e}")

if __name__ == "__main__":
    main()
