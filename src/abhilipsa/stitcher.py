import cv2
import numpy as np

class PanoramaStitcher:
    def make_panorama_for_images_in(self, images):
        
        # Preparing images for processing by converting to BGR
        bgr_images = [self._convert_to_bgr(image) for image in images]
        
        # Extracting features from each image
        keypoint_data, descriptor_data = self._extract_features_from_images(bgr_images)

        # Matching features across image pairs
        matches = self._match_descriptors(descriptor_data)

        # Calculating homographies for adjacent images
        homographies = self._compute_all_homographies(matches, keypoint_data)

        # Stitching images using OpenCV's default stitching pipeline
        panorama_result, stitching_status = self._stitch_images_with_opencv(bgr_images)

        if stitching_status:
            # Converting the stitched panorama back to RGB for display
            final_panorama = cv2.cvtColor(panorama_result, cv2.COLOR_BGR2RGB)
            return final_panorama, homographies
        else:
            print("Failed to generate panorama. Check image set.")
            return None, homographies

    def _convert_to_bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _extract_features_from_images(self, images):
        
        sift_detector = cv2.SIFT_create()
        keypoints, descriptors = [], []
        for image in images:
            kp, desc = sift_detector.detectAndCompute(image, None)
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors

    # def _match_descriptors(self, descriptors):
        
    #     flann_index_params = dict(algorithm=1, trees=5)
    #     flann_search_params = dict(checks=50)
    #     flann_matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    #     matched_pairs = []
    #     for i in range(len(descriptors) - 1):
    #         if descriptors[i] is not None and descriptors[i + 1] is not None:
    #             pairs = flann_matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)
    #             filtered_matches = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    #             matched_pairs.append(filtered_matches if len(filtered_matches) > 10 else [])
    #         else:
    #             matched_pairs.append([])
    #     return matched_pairs


    

def _match_descriptors(self, descriptors):
    matched_pairs = []
    
    # Iterate through each consecutive pair of descriptor sets
    for i in range(len(descriptors) - 1):
        if descriptors[i] is not None and descriptors[i + 1] is not None:
            matches = []

            # Iterate through each descriptor in the first set
            for j, desc1 in enumerate(descriptors[i]):
                distances = []
                
                # Calculate the Euclidean distance to each descriptor in the next image
                for k, desc2 in enumerate(descriptors[i + 1]):
                    distance = np.linalg.norm(desc1 - desc2)
                    distances.append((distance, k))  # Store distance and index of descriptor in img2

                # Sort distances to find the two closest matches
                distances = sorted(distances, key=lambda x: x[0])
                
                # Apply ratio test: the closest match should be significantly closer than the second
                if len(distances) > 1 and distances[0][0] < 0.75 * distances[1][0]:
                    best_match = cv2.DMatch(j, distances[0][1], distances[0][0])
                    matches.append(best_match)
            
            # Only retain matches if we have enough (e.g., more than 10)
            matched_pairs.append(matches if len(matches) > 10 else [])
        else:
            matched_pairs.append([])  # Append empty list if descriptors are missing
    return matched_pairs


    def _compute_all_homographies(self, matches, keypoints):
        """Calculate homographies for each matched image pair."""
        homography_matrices = []
        for i, match_set in enumerate(matches):
            if len(match_set) >= 4:
                src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match_set]).reshape(-1, 2)
                dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set]).reshape(-1, 2)
                homography = self._compute_single_homography(src_pts, dst_pts)
                homography_matrices.append(homography)
            else:
                homography_matrices.append(None)
        return homography_matrices

    def _compute_single_homography(self, src_pts, dst_pts):
        """Direct Linear Transform (DLT) to estimate a homography matrix."""
        A = []
        for (x, y), (xp, yp) in zip(src_pts, dst_pts):
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        
        _, _, Vt = np.linalg.svd(np.array(A))
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2] if H[2, 2] != 0 else None

    def _stitch_images_with_opencv(self, images):
        stitcher = cv2.Stitcher_create()
        status, stitched_img = stitcher.stitch(images)
        return stitched_img, (status == cv2.Stitcher_OK)
