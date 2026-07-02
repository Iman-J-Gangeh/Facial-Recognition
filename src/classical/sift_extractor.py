# src/classical/sift_extractor.py
# This file extracts SIFT features from images

import cv2
import numpy as np

from src.utils.exceptions import SIFTExtractionError


class SIFTExtractor:
    def __init__(self):
        # create SIFT object
        self.sift = cv2.SIFT_create()

    def extract_one(self, image):
        # make sure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # get keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        # if no features were found
        if descriptors is None or len(descriptors) == 0:
            return None

        return descriptors.astype(np.float32)

    def extract_all(self, samples):
        # collect descriptors from every image
        all_descriptors = []

        for sample in samples:
            try:
                descriptors = self.extract_one(sample.image)
            except Exception as e:
                print("Could not extract SIFT descriptors from", sample.source_path)
                print(e)
                continue

            # skip images with no descriptors
            if descriptors is None:
                print("No SIFT keypoints found in", sample.source_path)
                continue

            all_descriptors.append(descriptors)

        # if nothing was extracted, stop the program
        if len(all_descriptors) == 0:
            raise SIFTExtractionError(
                "No SIFT descriptors were found in any image."
            )

        # stack all descriptor arrays into one big array
        return np.vstack(all_descriptors)