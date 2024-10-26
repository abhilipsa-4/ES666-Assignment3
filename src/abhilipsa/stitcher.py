import cv2
import numpy as np
import os

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, images):
        # List to store the homography matrices
        homography_matrices = []

        # Use the first image as the base
        stitched_image = images[0]

        # Process each subsequent image
        for i in range(1, len(images)):
            # Detect and describe features in both images
            keypoints1, descriptors1 = self.detect_and_describe(stitched_image)
            keypoints2, descriptors2 = self.detect_and_describe(images[i])

            # Match features between the images
            matches = self.match_features(descriptors1, descriptors2)
            if len(matches) < 10:
                print(f"Not enough matches between images {i-1} and {i}. Skipping.")
                continue

            # Estimate homography matrix
            H = self.estimate_homography(keypoints1, keypoints2, matches)
            homography_matrices.append(H)

            # Warp and blend images together
            stitched_image = self.warp_and_blend(stitched_image, images[i], H)

        return stitched_image, homography_matrices

    def detect_and_describe(self, image):
        # Detect features and compute descriptors using SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2):
        # Match features using BFMatcher with SIFT descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        # Sort matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:50]  # Use the top 50 matches

    def estimate_homography(self, keypoints1, keypoints2, matches):
        # Extract matched points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate the homography matrix with RANSAC
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def warp_and_blend(self, image1, image2, H):
        # Dimensions for the output panorama
        height, width = image1.shape[:2]
        warped_image2 = cv2.warpPerspective(image2, H, (width + image2.shape[1], height))

        # Create a black canvas for the result with the size of the warped image
        result = np.zeros_like(warped_image2)
        result[0:height, 0:width] = image1  # Place the first image on the result

        # Define the mask for blending
        mask1 = (result[:, :, 0] > 0).astype(np.uint8)  # Non-zero regions of image1
        mask2 = (warped_image2[:, :, 0] > 0).astype(np.uint8)  # Non-zero regions of image2

        overlap = cv2.bitwise_and(mask1, mask2)  # Overlap region mask
        unique1 = mask1 - overlap  # Unique region of image1
        unique2 = mask2 - overlap  # Unique region of image2

        # Combine images by giving overlap regions an equal weight
        for c in range(3):  # For each color channel
            result[:, :, c] = (result[:, :, c] * unique1 +
                               warped_image2[:, :, c] * unique2 +
                               (result[:, :, c] // 2 + warped_image2[:, :, c] // 2) * overlap)

        return result

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def main():
    # List of folder names to process
    folders = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
    
    # Initialize the stitcher
    stitcher = PanaromaStitcher()
    
    # Loop through each folder to create panoramas
    for folder in folders:
        folder_path = f'Images/{folder}'
        
        # Load images from the current folder
        images = load_images_from_folder(folder_path)
        if len(images) < 2:
            print(f"Not enough images in {folder} to create a panorama. Skipping.")
            continue
        
        # Create panorama
        try:
            panorama, homographies = stitcher.make_panaroma_for_images_in(images)
            
            # Save the results
            output_path = f'./results/{folder}_panorama.jpg'
            if not os.path.exists('results'):
                os.makedirs('results')
            cv2.imwrite(output_path, panorama)
            print(f"Panorama created and saved for {folder}!")
        except Exception as e:
            print(f"Failed to create panorama for {folder}. Error: {e}")

if __name__ == "__main__":
    main()
