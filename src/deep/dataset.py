"""
src/deep/dataset.py

Dataset processing for the deep learning model.
"""

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.preprocessing.dataset_loader import Sample
from src.preprocessing.splitter import SplitIndex
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_label_mapping(samples):
    # Get all labels and give each one a number
    labels = []

    for sample in samples:
        if sample.label not in labels:
            labels.append(sample.label)

    labels.sort()

    label_to_idx = {}
    for i, label in enumerate(labels):
        label_to_idx[label] = i

    return label_to_idx


def get_train_transform(config):
    # Transforms used for training
    w, h = config.image_size

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((h, w), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.norm_mean, std=config.norm_std),
    ])

    return train_transform


def get_eval_transform(config):
    # Transforms used for validation and testing
    w, h = config.image_size

    eval_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.norm_mean, std=config.norm_std),
    ])

    return eval_transform


class FaceDataset(Dataset):
    def __init__(self, samples, transform, label_to_idx):
        self.samples = samples
        self.transform = transform
        self.label_to_idx = label_to_idx

        # Default image size in case an image fails to load
        self.num_channels = 3
        self.img_h = 224
        self.img_w = 224

        # Try to get image size from the transform
        for t in transform.transforms:
            if hasattr(t, "size"):
                size = t.size

                if type(size) == tuple or type(size) == list:
                    if len(size) == 2:
                        self.img_h = size[0]
                        self.img_w = size[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            img = Image.open(sample.path)
            img = img.convert("RGB")
            img = self.transform(img)

        except Exception as e:
            logger.warning("Could not load image %s. Using blank image instead.", sample.path)

            img = torch.zeros(
                self.num_channels,
                self.img_h,
                self.img_w,
                dtype=torch.float32
            )

        if sample.label in self.label_to_idx:
            label = self.label_to_idx[sample.label]
        else:
            label = 0

        return img, label


def make_dataloaders(split_index, config):
    # Combine all samples so labels are consistent across train/val/test
    all_samples = split_index.train + split_index.val + split_index.test
    label_to_idx = build_label_mapping(all_samples)

    train_transform = get_train_transform(config)
    eval_transform = get_eval_transform(config)

    train_dataset = FaceDataset(split_index.train, train_transform, label_to_idx)
    val_dataset = FaceDataset(split_index.val, eval_transform, label_to_idx)
    test_dataset = FaceDataset(split_index.test, eval_transform, label_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(
        "Created dataloaders: train=%d, val=%d, test=%d, classes=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        len(label_to_idx)
    )

    return train_loader, val_loader, test_loader, label_to_idx


def get_deep_dataset(config, split_index, preprocessor, mode="train_val"):
    # preprocessor is not used here, but kept so the function matches the rest of the project
    train_loader, val_loader, test_loader, label_to_idx = make_dataloaders(
        split_index,
        config
    )

    if mode == "train":
        return train_loader, None, label_to_idx

    if mode == "test":
        return None, test_loader, label_to_idx

    if mode == "train_val":
        return train_loader, val_loader, label_to_idx

    raise ValueError("Invalid mode: " + mode)