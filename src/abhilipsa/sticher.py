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
            # Detect and describe features in both images
            keypoints1, descriptors1 = self.detect_and_describe(stitched_image)
            keypoints2, descriptors2 = self.detect_and_describe(images[i])

            # Match features between the images
            matches = self.match_features(descriptors1, descriptors2)

            # Estimate homography matrix
            H = self.estimate_homography(keypoints1, keypoints2, matches)
            homography_matrices.append(H)

            # Warp and blend images together
            stitched_image = self.warp_and_blend(stitched_image, images[i], H)

        return stitched_image, homography_matrices

    def detect_and_describe(self, image):
        # Detect features and compute descriptors using ORB
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2):
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
        # Warp image2 to image1's perspective
        width = image1.shape[1] + image2.shape[1]
        height = image1.shape[0]
        result = cv2.warpPerspective(image2, H, (width, height))
        # Blend the images
        result[0:image1.shape[0], 0:image1.shape[1]] = image1
        return result

