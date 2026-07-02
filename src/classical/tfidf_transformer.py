# src/classical/tfidf_transformer.py
# This file applies TF-IDF weighting to Bag of Visual Words histograms

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer as SklearnTFIDF

from src.utils.artifact_manager import ArtifactManager
from src.utils.exceptions import ArtifactError


artifact_manager = ArtifactManager()


class TFIDFTransformer:
    def __init__(self, config):
        self.config = config
        self.transformer = None

    def fit_transform(self, X):
        # create the TF-IDF transformer
        self.transformer = SklearnTFIDF(
            smooth_idf=True,
            sublinear_tf=False
        )

        # fit only on training data
        self.transformer.fit(X)

        # transform the data and convert it back to a numpy array
        result = self.transformer.transform(X)
        result = result.toarray().astype(np.float32)

        # save the transformer
        metadata = {
            "vocab_size": X.shape[1]
        }

        try:
            folder = Path(self.config.tfidf_artifact_path).parent
            folder.mkdir(parents=True, exist_ok=True)

            artifact_manager.save(
                self.transformer,
                self.config.tfidf_artifact_path,
                metadata
            )

            print("Saved TF-IDF transformer to", self.config.tfidf_artifact_path)

        except Exception as e:
            raise ArtifactError(
                "Could not save TF-IDF transformer: " + str(e)
            )

        return result

    def transform(self, X):
        # make sure the transformer has already been fitted
        if self.transformer is None:
            raise RuntimeError(
                "TFIDFTransformer has not been fitted yet."
            )

        result = self.transformer.transform(X)
        result = result.toarray().astype(np.float32)

        return result

    def load_or_fit(self, X_train):
        path = self.config.tfidf_artifact_path

        # try loading the saved transformer first
        if Path(path).exists() and not self.config.retrain:
            try:
                self.transformer = artifact_manager.load(
                    path,
                    expected_meta={
                        "vocab_size": X_train.shape[1]
                    }
                )

                print("Loaded TF-IDF transformer from", path)
                return self.transform(X_train)

            except Exception as e:
                print("Could not load saved TF-IDF transformer, fitting again...")
                print(e)

        # fit if there was no saved transformer or loading failed
        return self.fit_transform(X_train)
