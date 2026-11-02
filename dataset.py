import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

CATEGORIES = ['CurbRamp', 'SurfaceProblem', 'Obstacle', 'Crosswalk']
META_COLUMNS = ['filename', 'normalized_x', 'normalized_y']

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_STATS = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
NORMALIZATION = {'cnn': IMAGENET_STATS, 'resnet50': IMAGENET_STATS, 'dinov2': IMAGENET_STATS, 'clip': CLIP_STATS}


@dataclass
class Samples:
    paths: list
    labels: np.ndarray      # (N, num_tags) multi-hot
    categories: np.ndarray  # (N,) index into CATEGORIES

    def __len__(self):
        return len(self.paths)

    def select(self, indices):
        return Samples([self.paths[i] for i in indices], self.labels[indices], self.categories[indices])


def build_vocabulary(data_dir):
    """Tag names shared across categories (e.g. bumpy) map to one output of the head."""
    category_tags = {}
    for category in CATEGORIES:
        header = pd.read_csv(os.path.join(data_dir, category, 'train.csv'), nrows=0).columns
        category_tags[category] = [name for name in header if name not in META_COLUMNS]
    tags = sorted({tag for names in category_tags.values() for tag in names})
    return tags, category_tags


def build_category_masks(tags, category_tags):
    """masks[c, t] is True when tag t is valid for images of category c."""
    masks = np.zeros((len(CATEGORIES), len(tags)), dtype=bool)
    for c, category in enumerate(CATEGORIES):
        for tag in category_tags[category]:
            masks[c, tags.index(tag)] = True
    return masks


def load_samples(data_dir, split, tags, category_tags, check_images=True):
    paths, labels, categories = [], [], []
    for c, category in enumerate(CATEGORIES):
        frame = pd.read_csv(os.path.join(data_dir, category, f'{split}.csv'))
        image_dir = os.path.join(data_dir, category, split)
        frame['path'] = [os.path.join(image_dir, name) for name in frame['filename']]

        if check_images:
            exists = frame['path'].map(os.path.exists)
            if not exists.all():
                print(f'{category}/{split}: {(~exists).sum()} of {len(frame)} images missing, skipping them')
            frame = frame[exists]

        category_labels = np.zeros((len(frame), len(tags)), dtype=np.float32)
        for tag in category_tags[category]:
            category_labels[:, tags.index(tag)] = frame[tag].to_numpy(dtype=np.float32)

        paths.extend(frame['path'])
        labels.append(category_labels)
        categories.append(np.full(len(frame), c, dtype=np.int64))
    return Samples(paths, np.concatenate(labels), np.concatenate(categories))


def train_val_split(samples, val_fraction=0.1, seed=42):
    indices = np.arange(len(samples))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=samples.categories)
    return samples.select(train_idx), samples.select(val_idx)


def build_transform(backbone, image_size, train=False):
    mean, std = NORMALIZATION[backbone]
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2)]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class SidewalkDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples.paths[idx]).convert('RGB')
        return self.transform(image), torch.from_numpy(self.samples.labels[idx]), int(self.samples.categories[idx])


def compute_pos_weight(samples, category_masks):
    """neg/pos per tag, counted only on images where the tag is valid. Capped so rare tags don't blow up the loss."""
    valid = category_masks[samples.categories]
    positives = (samples.labels * valid).sum(axis=0)
    negatives = valid.sum(axis=0) - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)
    return torch.tensor(np.clip(weights, 1.0, 50.0), dtype=torch.float32)


def rare_tag_sampler(samples):
    """Oversample images by the rarity of their rarest tag. Images with no tags get the lowest weight."""
    counts = samples.labels.sum(axis=0)
    tag_weights = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    weights = (samples.labels * tag_weights).max(axis=1)
    lowest = tag_weights[tag_weights > 0].min()
    weights = np.maximum(weights, lowest)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(samples), replacement=True)
