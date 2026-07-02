# src/classical/verification_engine.py
# This file compares two sets of face images and decides if they match

import cv2
import numpy as np

from src.classical.sift_extractor import SIFTExtractor
from src.utils.exceptions import VerificationError


RANSAC_ALPHA = 0.7


class VerificationResult:
    def __init__(self, score, match, ransac_inlier_ratio=None):
        self.score = score
        self.match = match
        self.ransac_inlier_ratio = ransac_inlier_ratio


class VerificationEngine:
    def __init__(self, bovw_encoder, tfidf_transformer, config):
        self.encoder = bovw_encoder
        self.tfidf = tfidf_transformer
        self.config = config
        self.sift = SIFTExtractor()

    def verify(self, set_a, set_b):
        # make sure both sets have images
        if len(set_a) == 0:
            raise VerificationError("Set A is empty.")

        if len(set_b) == 0:
            raise VerificationError("Set B is empty.")

        # get average TF-IDF vectors for both image sets
        avg_a = self.compute_mean_tfidf(set_a, "A")
        avg_b = self.compute_mean_tfidf(set_b, "B")

        # compare the two vectors
        distance = self.compute_distance(avg_a, avg_b)

        # change distance into a similarity score
        bovw_score = 1.0 / (1.0 + distance)

        ransac_ratio = None

        # optionally use RANSAC too
        if self.config.ransac_enabled:
            ransac_ratio = self.ransac_inlier_ratio(set_a, set_b)

            if ransac_ratio is not None:
                bovw_score = (
                    RANSAC_ALPHA * bovw_score
                    + (1.0 - RANSAC_ALPHA) * ransac_ratio
                )

        # keep score between 0 and 1
        final_score = float(np.clip(bovw_score, 0.0, 1.0))

        # decide if it is a match
        match = final_score >= self.config.verification_threshold

        return VerificationResult(final_score, match, ransac_ratio)

    def compute_mean_tfidf(self, samples, set_name):
        histograms = []

        for sample in samples:
            descriptors = self.sift.extract_one(sample.image)
            hist = self.encoder.encode(descriptors)
            histograms.append(hist)

        if len(histograms) == 0:
            raise VerificationError(
                "No usable images found in set " + set_name
            )

        # stack histograms and apply TF-IDF
        hist_matrix = np.vstack(histograms)
        tfidf_matrix = self.tfidf.transform(hist_matrix)

        # average all image vectors into one vector
        return np.mean(tfidf_matrix, axis=0)

    def compute_distance(self, vec_a, vec_b):
        metric = self.config.knn_metric.lower()

        if metric == "cosine":
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a == 0 or norm_b == 0:
                return 1.0

            similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            return 1.0 - float(similarity)

        # default distance is Euclidean
        return float(np.linalg.norm(vec_a - vec_b))

    def ransac_inlier_ratio(self, set_a, set_b):
        try:
            # just use the first image from each set
            img_a = set_a[0].image
            img_b = set_b[0].image

            # make images grayscale if needed
            if len(img_a.shape) == 3:
                img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)

            if len(img_b.shape) == 3:
                img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()

            kp_a, desc_a = sift.detectAndCompute(img_a, None)
            kp_b, desc_b = sift.detectAndCompute(img_b, None)

            # need enough keypoints for homography
            if desc_a is None or desc_b is None:
                print("RANSAC skipped because descriptors were missing.")
                return None

            if len(kp_a) < 4 or len(kp_b) < 4:
                print("RANSAC skipped because there were not enough keypoints.")
                return None

            # match descriptors
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)

            good_matches = []

            for match_pair in raw_matches:
                if len(match_pair) != 2:
                    continue

                m, n = match_pair

                # Lowe's ratio test
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                print("RANSAC skipped because there were too few good matches.")
                return None

            points_a = []
            points_b = []

            for match in good_matches:
                points_a.append(kp_a[match.queryIdx].pt)
                points_b.append(kp_b[match.trainIdx].pt)

            points_a = np.float32(points_a).reshape(-1, 1, 2)
            points_b = np.float32(points_b).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(
                points_a,
                points_b,
                cv2.RANSAC,
                5.0
            )

            if mask is None:
                return None

            inliers = mask.ravel().sum()
            total = len(mask)

            ratio = float(inliers) / total
            return ratio

        except Exception as e:
            print("RANSAC failed, so it was skipped.")
            print(e)
            return None
