import cv2
import numpy as np

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, images):
        homography_matrices = []
        stitched_image = images[0]

        for i in range(1, len(images)):
            keypoints1, descriptors1 = self.detect_and_describe(stitched_image)
            keypoints2, descriptors2 = self.detect_and_describe(images[i])

            matches = self.match_features(descriptors1, descriptors2)
            print(f"Number of matches between images {i-1} and {i}: {len(matches)}")

            if len(matches) < 10:
                print(f"Not enough matches for stitching images {i-1} and {i}. Skipping.")
                continue

            H = self.estimate_homography(keypoints1, keypoints2, matches)
            if H is None:
                print(f"Failed to estimate homography between images {i-1} and {i}. Skipping.")
                continue
            homography_matrices.append(H)

            stitched_image = self.warp_and_blend(stitched_image, images[i], H)
            if stitched_image is None:
                print(f"Failed to warp and blend images {i-1} and {i}.")
                break

        return stitched_image, homography_matrices

    def detect_and_describe(self, image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        return good_matches

    def estimate_homography(self, keypoints1, keypoints2, matches):
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        if len(src_pts) >= 4:
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        else:
            return None

    def warp_and_blend(self, image1, image2, H):
        height, width = image1.shape[:2]
        warped_image2 = cv2.warpPerspective(image2, H, (width + image2.shape[1], height))

        if warped_image2 is None:
            print("Warping failed.")
            return None

        result = np.zeros_like(warped_image2)
        result[:height, :width] = image1

        mask1 = (result > 0).astype(float)
        mask2 = (warped_image2 > 0).astype(float)
        blended_result = (result * mask1 + warped_image2 * mask2) / (mask1 + mask2 + 1e-8)
        blended_result = np.clip(blended_result, 0, 255).astype(np.uint8)

        return blended_result
