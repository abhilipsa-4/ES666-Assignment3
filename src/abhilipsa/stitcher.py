import cv2
import numpy as np
import os

class PanoramaStitcher:
    def __init__(self):
        # Initialize a list to store homography matrices between image pairs
        self.homographies = []

    def find_homography(self, img1, img2):
        # Detect ORB features and compute descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # Match descriptors using the BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance for robust matching
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Select good matches (for example, the top 10%)
        num_good_matches = int(len(matches) * 0.1)
        matches = matches[:num_good_matches]

        # Extract matched points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography matrix using RANSAC
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def make_panorama_for_images_in(self, folder_path):
        images = []

        # Load images from the specified folder
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)

        # Stitch images using calculated homographies
        base_img = images[0]
        stitched_image = base_img
        for i in range(1, len(images)):
            H = self.find_homography(stitched_image, images[i])
            self.homographies.append(H)
            
            # Warp image based on homography
            height, width = stitched_image.shape[:2]
            warped_image = cv2.warpPerspective(images[i], H, (width * 2, height))
            
            # Update stitched image (you can implement blending if needed)
            stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_image, 0.5, 0)

        return stitched_image, self.homographies
