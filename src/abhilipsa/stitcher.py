import cv2
import numpy as np

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, images):
        # List to store the homography matrices
        homography_matrices = []

        # Use the first image as the base
        stitched_image = images[0]

        # Process each subsequent image
        for i in range(1, len(images)):
            # Detect and describe features in both images using SIFT
            keypoints1, descriptors1 = self.detect_and_describe(stitched_image)
            keypoints2, descriptors2 = self.detect_and_describe(images[i])

            # Match features between the images
            matches = self.match_features(descriptors1, descriptors2)

            # Check for sufficient matches
            if len(matches) < 10:
                print(f"Not enough matches found between images {i-1} and {i}. Skipping.")
                continue

            # Estimate homography matrix from scratch
            H = self.estimate_homography(keypoints1, keypoints2, matches)
            if H is None:
                print(f"Failed to estimate homography between images {i-1} and {i}. Skipping.")
                continue
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
        # Match features using BFMatcher with default parameters
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches

    def estimate_homography(self, keypoints1, keypoints2, matches):
        # Extract matched points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Calculate homography matrix from scratch using least squares
        if len(src_pts) >= 4:  # Minimum 4 points for homography
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        else:
            return None

    def warp_and_blend(self, image1, image2, H):
        # Warp image2 to image1's perspective
        height, width = image1.shape[:2]
        warped_image2 = cv2.warpPerspective(image2, H, (width + image2.shape[1], height))

        # Create a blank canvas for the final panorama
        result = np.zeros_like(warped_image2)
        result[:height, :width] = image1

        # Blend images in the overlapping region
        mask1 = (result > 0).astype(float)
        mask2 = (warped_image2 > 0).astype(float)
        blended_result = (result * mask1 + warped_image2 * mask2) / (mask1 + mask2 + 1e-8)
        blended_result = np.clip(blended_result, 0, 255).astype(np.uint8)

        return blended_result
