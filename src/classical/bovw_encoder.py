# src/classical/bovw_encoder.py
# This file converts SIFT descriptors into Bag of Visual Words histograms

import numpy as np


class BoVWEncoder:
    def __init__(self, vocabulary):
        # save the vocabulary and the cluster centers
        self.vocab = vocabulary
        self.centers = vocabulary.centers
        self.vocab_size = vocabulary.vocab_size

    def encode(self, descriptors):
        # make an empty histogram
        hist = np.zeros(self.vocab_size, dtype=np.float32)

        # if the image has no descriptors, return all zeros
        if descriptors is None or len(descriptors) == 0:
            return hist

        # find the closest visual word for each descriptor
        for desc in descriptors:
            distances = np.sum((self.centers - desc) ** 2, axis=1)
            closest_word = np.argmin(distances)
            hist[closest_word] += 1

        # normalize the histogram
        norm = np.linalg.norm(hist)
        if norm != 0:
            hist = hist / norm

        return hist

    def encode_batch(self, samples, extractor):
        # encode all images one by one
        features = []

        for sample in samples:
            descriptors = extractor.extract_one(sample.image)
            hist = self.encode(descriptors)
            features.append(hist)

        if len(features) == 0:
            return np.empty((0, self.vocab_size), dtype=np.float32)

        return np.vstack(features)