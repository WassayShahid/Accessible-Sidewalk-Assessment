# Accessible Sidewalk Assessment Using Multi-Label Classification

CS/SE 4AL3 project, Group 33. A single model that takes a Street View crop of a sidewalk feature and its category, and predicts which accessibility conditions are present (cracks, steep, pole, and so on).

Adapted from the [CIFAR-10 classifier](https://github.com/WassayShahid/Classifier). Nothing has been trained yet, so this repo has no results.

## Setup

Python 3.10 or newer. Install the PyTorch build for your GPU from [pytorch.org](https://pytorch.org/get-started/locally/) first, then:

```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

The smoke test runs the whole pipeline on a tiny fake dataset. It needs no real data or pretrained weights.

## Data

We use the cleaned dataset from [sidewalk-tagger-ai](https://github.com/ProjectSidewalk/sidewalk-tagger-ai), hosted on [Hugging Face](https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated).

The four zips in `Validated/` total about 71 GB, and the images inside are uncropped. Unzip them so you have `data/raw/CurbRamp/crops`, `data/raw/CurbRamp/csv`, and the same for the other three categories, then crop:

```bash
python prepare_data.py
```

This writes 640x640 crops to `data/processed/`, using the same crop as the authors' `crop.py`. Labels near an image edge give a smaller crop. Once it finishes, `data/raw/` can be deleted.

| Category | Train | Test | Tag columns |
| --- | --- | --- | --- |
| CurbRamp | 8,674 | 2,183 | 10 |
| SurfaceProblem | 7,282 | 1,810 | 13 |
| Obstacle | 1,954 | 478 | 20 |
| Crosswalk | 1,306 | 321 | 10 |

10% of the training set is held out for validation, stratified by category.

## Training

```bash
# ResNet-50 baseline
python train.py --backbone resnet50

# Long-tail methods
python train.py --backbone resnet50 --loss weighted_bce
python train.py --backbone resnet50 --loss focal
python train.py --backbone resnet50 --oversample

# Vision transformers: linear probing and full fine-tuning
python train.py --backbone dinov2 --linear-probe
python train.py --backbone dinov2
python train.py --backbone clip --linear-probe
python train.py --backbone clip
```

Each run keeps the epoch with the best validation macro F1 in `checkpoints/`, along with thresholds tuned on the validation set. Lower `--batch-size` if you run out of GPU memory.

## Evaluation and prediction

```bash
python evaluate.py checkpoints/resnet50-bce.pth
python predict.py checkpoints/resnet50-bce.pth --category CurbRamp --image crop.png
```

`evaluate.py` prints per-tag precision, recall and F1, then macro, micro and weighted F1 overall and per category. It also saves the per-tag table to `results/`. Leave out `--image` in `predict.py` to open a file picker.

## How the model works

- **Unified model:** one shared backbone and one sigmoid output per tag name.
- **Category mask:** each image's category decides which tags are valid. The loss ignores invalid tags, and at inference they are forced to zero.
- **Thresholds:** one threshold per (category, tag) pair, tuned on validation for F1.

### Number of tags

The proposal says 33 tags, but the dataset CSVs have 53 category/tag columns, which come to 46 unique tag names. Seven names appear in two categories (`narrow`, `bumpy`, `brick-cobblestone`, `construction`, `height-difference`, `rail-tram-track`, `uneven-slanted`).

33 is how many pairs have at least 10 positives in the test set. So 33 is a reporting cutoff, not the size of the output layer:

- The head has 46 outputs, so shared tags share one output, as the proposal intends.
- `evaluate.py` scores each (category, tag) pair separately. The `Benchmark` line covers the 33 pairs and is the one to compare with Liu et al. The line below it covers every pair with a test positive.

Four outputs can't learn anything because their tags have no positives in the training set: `parallel-lines`, `level-with-sidewalk`, `no-pedestrian-priority` and `too-close-to-traffic`. None of the four is in the 33.

## Changes from the CIFAR-10 code

| CIFAR-10 classifier | This repo |
| --- | --- |
| `CrossEntropyLoss`, softmax, top-1 prediction | Masked BCE or focal loss, sigmoid, tuned thresholds |
| Accuracy, `classification_report` | Per-tag precision, recall and F1, plus macro, micro and weighted F1 |
| Custom ResNet with a 3x3 first convolution for 32x32 images | torchvision ResNet-50 with ImageNet weights, plus DINOv2 and CLIP through timm |
| SGD at one learning rate | AdamW, with a lower rate for the backbone than the head |
| `torchvision.datasets.CIFAR10` | `SidewalkDataset`, reading the four category CSVs |
| Random crop and flip also applied to the test set | Augmentation on training data only |
| Test set used for validation, last epoch saved | Separate validation split, best epoch saved |
| CIFAR-10 normalisation, inconsistent between train and predict | Normalisation matched to each backbone, shared by train and predict |

The `CosineAnnealingLR` schedule, dropout of 0.25 and file picker carry over.

## Not implemented

Holding out one city to test generalisation, which the proposal lists as "if time permits". The city is the second part of each filename (`gsv-seattle-...`).

## References

X. Liu et al. Towards Fine-Grained Sidewalk Accessibility Assessment with Deep Learning: Initial Benchmarks and an Open Dataset. ASSETS, 2024. https://doi.org/10.1145/3663548.3688531
