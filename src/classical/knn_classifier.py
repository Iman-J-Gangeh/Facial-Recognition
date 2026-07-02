# src/classical/knn_classifier.py
# KNN classifier for the Bag of Visual Words project

from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.utils.artifact_manager import ArtifactManager
from src.utils.exceptions import ArtifactError, ConfigError


artifact_manager = ArtifactManager()


class KNNClassifier:
    def __init__(self, config):
        self.config = config
        self.clf = None
        self.classes = []

    def train(self, X, y):
        # make sure k is valid
        if self.config.knn_k < 1:
            raise ConfigError("knn_k must be at least 1")

        print("Training KNN classifier...")
        print("k =", self.config.knn_k)
        print("metric =", self.config.knn_metric)
        print("number of samples =", len(y))

        # create and train the KNN model
        self.clf = KNeighborsClassifier(
            n_neighbors=self.config.knn_k,
            metric=self.config.knn_metric
        )

        self.clf.fit(X, y)
        self.classes = sorted(set(y))

        # save the trained model
        metadata = {
            "knn_k": self.config.knn_k,
            "knn_metric": self.config.knn_metric
        }

        try:
            folder = Path(self.config.knn_artifact_path).parent
            folder.mkdir(parents=True, exist_ok=True)

            artifact_manager.save(
                {
                    "clf": self.clf,
                    "classes": self.classes
                },
                self.config.knn_artifact_path,
                metadata
            )

            print("Saved KNN classifier to", self.config.knn_artifact_path)

        except Exception as e:
            raise ArtifactError("Could not save KNN classifier: " + str(e))

    def predict_topk(self, X, k=5):
        # check that the model has already been trained
        if self.clf is None:
            raise RuntimeError("KNNClassifier has not been trained yet.")

        # do not ask for more neighbors than we have training samples
        actual_k = min(k, self.clf.n_samples_fit_)

        # get nearest neighbors
        distances, indices = self.clf.kneighbors(X, n_neighbors=actual_k)

        train_labels = np.array(self.clf._y)
        all_predictions = []

        for row in indices:
            labels = []

            for i in row:
                label_index = train_labels[i]
                label = self.clf.classes_[label_index]
                labels.append(label)

            # count labels and sort by most common
            counts = Counter(labels)

            ranked_labels = sorted(
                counts.keys(),
                key=lambda label: (-counts[label], labels.index(label))
            )

            all_predictions.append(ranked_labels[:k])

        return all_predictions

    def load_or_train(self, X, y):
        path = self.config.knn_artifact_path

        # try loading the saved classifier first
        if Path(path).exists() and not self.config.retrain:
            try:
                payload = artifact_manager.load(
                    path,
                    expected_meta={
                        "knn_k": self.config.knn_k,
                        "knn_metric": self.config.knn_metric
                    }
                )

                self.clf = payload["clf"]
                self.classes = payload.get("classes", [])

                print("Loaded KNN classifier from", path)
                return self

            except Exception as e:
                print("Could not load saved KNN classifier, retraining...")
                print(e)

        # train if there was no saved model or loading failed
        self.train(X, y)
        return self