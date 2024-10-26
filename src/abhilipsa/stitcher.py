import cv2
import numpy as np

class PanoramaStitcher:
    def create_panorama(self, images):
        # Convert images from RGB to BGR format for OpenCV
        images_bgr = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]

        # Detect features and compute descriptors
        keypoints, descriptors = self.extract_features(images_bgr)

        # Match features between adjacent images
        matched_features = self.find_feature_matches(descriptors)

        # Calculate homography matrices for image alignment
        homographies = self.calculate_homographies(matched_features, keypoints)

        # Use OpenCV's built-in Stitcher to create the final panorama
        stitcher = cv2.Stitcher_create()
        status, final_image = stitcher.stitch(images_bgr)

        if status == cv2.Stitcher_OK:
            # Convert the stitched image back to RGB format
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            return final_image_rgb, homographies  # Return both the final image and homographies
        else:
            print("Error: Unable to create the panorama.")
            return None, homographies

    def extract_features(self, images):
        # Initialize SIFT for feature extraction
        sift = cv2.SIFT_create()
        kp_list = []
        desc_list = []
        for image in images:
            keypoints, descriptors = sift.detectAndCompute(image, None)
            kp_list.append(keypoints)
            desc_list.append(descriptors)
        return kp_list, desc_list

    def find_feature_matches(self, descriptors):
        # Set up the FLANN matcher
        flann_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(flann_params, search_params)

        matches = []
        for i in range(len(descriptors) - 1):
            if descriptors[i] is not None and descriptors[i + 1] is not None:
                knn_matches = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)

                # Apply Lowe's ratio test
                good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

                if len(good_matches) > 10:
                    matches.append(good_matches)
                else:
                    print(f"Insufficient matches between images {i} and {i + 1}. Skipping.")
                    matches.append([])  # Append an empty list for consistency
        return matches

    def calculate_homographies(self, matches, keypoints):
        homographies = []
        for i, match_set in enumerate(matches):
            if len(match_set) < 4:
                print(f"Not enough matches to compute homography for image pair {i}. Skipping.")
                homographies.append(None)
                continue

            src_points = np.float32([keypoints[i][m.queryIdx].pt for m in match_set]).reshape(-1, 2)
            dst_points = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set]).reshape(-1, 2)

            # Calculate the homography matrix using Direct Linear Transform
            H_matrix = self.compute_homography(src_points, dst_points)
            if H_matrix is not None:
                homographies.append(H_matrix)
            else:
                homographies.append(None)

        return homographies

    def compute_homography(self, src_pts, dst_pts):
        # Construct the matrix for DLT
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            x_prime, y_prime = dst_pts[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2] if H[2, 2] != 0 else None
