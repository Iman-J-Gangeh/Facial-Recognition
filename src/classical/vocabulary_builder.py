# src/classical/vocabulary_builder.py
# This file builds the visual vocabulary using K-means

from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.utils.artifact_manager import ArtifactManager
from src.utils.exceptions import ArtifactError, VocabularyError


artifact_manager = ArtifactManager()


class KMeansVocabulary:
    def __init__(self, centers, vocab_size):
        self.centers = centers
        self.vocab_size = vocab_size


class VocabularyBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, descriptors):
        # make sure we actually have descriptors
        if descriptors is None or len(descriptors) == 0:
            raise VocabularyError(
                "Cannot build vocabulary because there are no descriptors."
            )

        print("Building visual vocabulary...")
        print("vocab size =", self.config.vocab_size)
        print("number of descriptors =", len(descriptors))

        # create K-means model
        kmeans = MiniBatchKMeans(
            n_clusters=self.config.vocab_size,
            random_state=self.config.random_seed,
            max_iter=self.config.kmeans_max_iter,
            n_init=3,
            batch_size=min(10 * self.config.vocab_size, len(descriptors))
        )

        # train K-means on the SIFT descriptors
        try:
            kmeans.fit(descriptors)
        except Exception as e:
            print("K-means had a problem while training.")
            print(e)
            print("Trying to use the current result anyway.")

        # save the cluster centers as the vocabulary
        vocab = KMeansVocabulary(
            centers=kmeans.cluster_centers_.astype(np.float32),
            vocab_size=self.config.vocab_size
        )

        # save vocabulary to disk
        metadata = {
            "vocab_size": self.config.vocab_size,
            "random_seed": self.config.random_seed
        }

        try:
            folder = Path(self.config.bovw_artifact_path).parent
            folder.mkdir(parents=True, exist_ok=True)

            artifact_manager.save(
                vocab,
                self.config.bovw_artifact_path,
                metadata
            )

            print("Saved vocabulary to", self.config.bovw_artifact_path)

        except Exception as e:
            raise ArtifactError(
                "Could not save vocabulary: " + str(e)
            )

        return vocab

    def load_or_build(self, descriptors):
        path = self.config.bovw_artifact_path

        # try loading saved vocabulary first
        if Path(path).exists() and not self.config.retrain:
            try:
                vocab = artifact_manager.load(
                    path,
                    expected_meta={
                        "vocab_size": self.config.vocab_size
                    }
                )

                print("Loaded vocabulary from", path)
                return vocab

            except Exception as e:
                print("Could not load saved vocabulary, building it again...")
                print(e)

        # build vocabulary if loading failed or retrain is true
        return self.build(descriptors)
