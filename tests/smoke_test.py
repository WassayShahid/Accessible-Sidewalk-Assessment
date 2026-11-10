"""Checks the pipeline on a tiny fake dataset, without real data or pretrained weights.

Run from the repo root: python tests/smoke_test.py
"""
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate
import prepare_data
import train
from dataset import CATEGORIES, build_category_masks, build_vocabulary, compute_pos_weight, load_samples
from losses import MaskedBCELoss, MaskedFocalLoss
from metrics import apply_thresholds, per_tag_metrics, summarize, tune_thresholds
from model import ConvNeuralNet, SidewalkTagger, load_checkpoint
from predict import predict_tags

TAGS = {
    'CurbRamp': ['narrow', 'steep'],
    'SurfaceProblem': ['bumpy', 'cracks'],
    'Obstacle': ['narrow', 'pole'],
    'Crosswalk': ['bumpy', 'paint-fading'],
}


def write_raw_dataset(raw_dir):
    rng = np.random.default_rng(0)
    for category in CATEGORIES:
        os.makedirs(os.path.join(raw_dir, category, 'crops'))
        os.makedirs(os.path.join(raw_dir, category, 'csv'))
        for split, count in (('train', 12), ('test', 4)):
            rows = []
            for i in range(count):
                filename = f'gsv-testcity-{split}{i}-{category}.png'
                pixels = rng.integers(0, 255, (720, 960, 4), dtype=np.uint8)  # the real images are RGBA PNGs
                Image.fromarray(pixels).save(os.path.join(raw_dir, category, 'crops', filename))
                row = {'filename': filename, 'normalized_x': 0.5, 'normalized_y': 0.5}
                row.update({tag: int(rng.integers(0, 2)) for tag in TAGS[category]})
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(raw_dir, category, 'csv', f'{split}.csv'), index=False)


def test_crop(tmp):
    source = os.path.join(tmp, 'wide.png')
    Image.new('RGB', (1000, 800)).save(source)
    centre, corner = os.path.join(tmp, 'centre.png'), os.path.join(tmp, 'corner.png')
    prepare_data.crop_image(source, centre, 0.5, 0.5)
    prepare_data.crop_image(source, corner, 0.1, 0.1)
    assert Image.open(centre).size == (640, 640)
    assert Image.open(corner).size == (420, 400)


def test_zip_source(tmp, raw_dir):
    """Cropping straight from the zips, saved as JPEG, gives the same samples as the unzipped PNG route."""
    zip_dir, jpeg_dir = os.path.join(tmp, 'zips'), os.path.join(tmp, 'processed-jpeg')
    os.makedirs(zip_dir)
    for category in CATEGORIES:
        with zipfile.ZipFile(os.path.join(zip_dir, f'{category}.zip'), 'w') as archive:
            for folder, _, files in os.walk(os.path.join(raw_dir, category)):
                for name in files:
                    path = os.path.join(folder, name)
                    archive.write(path, os.path.relpath(path, raw_dir).replace(os.sep, '/'))
    prepare_data.main(['--zip-dir', zip_dir, '--output-dir', jpeg_dir, '--jpeg'])
    prepare_data.main(['--zip-dir', zip_dir, '--output-dir', jpeg_dir, '--jpeg', '--categories', 'Obstacle'])  # rerun skips

    tags, category_tags = build_vocabulary(jpeg_dir)
    samples = load_samples(jpeg_dir, 'train', tags, category_tags)
    assert len(samples) == 48 and all(path.endswith('.jpg') for path in samples.paths)
    with Image.open(samples.paths[0]) as image:
        assert image.format == 'JPEG' and image.size == (640, 640)
    return jpeg_dir


def test_resume(tmp, data_dir):
    output_dir = os.path.join(tmp, 'resume')
    common = ['--data-dir', data_dir, '--output-dir', output_dir, '--no-pretrained', '--backbone', 'resnet50',
              '--batch-size', '4', '--image-size', '64', '--num-workers', '0', '--resume']
    train.main(common + ['--epochs', '1'])
    last_path = os.path.join(output_dir, 'resnet50-bce-last.pth')
    assert torch.load(last_path)['epoch'] == 1
    train.main(common + ['--epochs', '2'])
    assert torch.load(last_path)['epoch'] == 2
    train.main(common + ['--epochs', '2'])  # already finished: trains nothing
    assert torch.load(last_path)['epoch'] == 2
    assert os.path.exists(os.path.join(output_dir, 'resnet50-bce.pth'))


def test_vocabulary(data_dir):
    tags, category_tags = build_vocabulary(data_dir)
    masks = build_category_masks(tags, category_tags)
    assert tags == ['bumpy', 'cracks', 'narrow', 'paint-fading', 'pole', 'steep']
    assert masks.shape == (4, 6) and masks.sum() == 8
    assert masks[CATEGORIES.index('CurbRamp'), tags.index('narrow')]
    assert masks[CATEGORIES.index('Obstacle'), tags.index('narrow')]
    assert not masks[CATEGORIES.index('Crosswalk'), tags.index('steep')]

    samples = load_samples(data_dir, 'train', tags, category_tags)
    assert len(samples) == 48
    assert not (samples.labels * ~masks[samples.categories]).any()
    assert compute_pos_weight(samples, masks).shape == (6,)


def test_losses_ignore_invalid_tags():
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    masks = torch.tensor([[1.0, 1.0, 0.0]])
    for criterion in (MaskedBCELoss(), MaskedBCELoss(torch.ones(3)), MaskedFocalLoss()):
        logits = torch.tensor([[0.5, -0.5, 9.0]], requires_grad=True)
        criterion(logits, labels, masks).backward()
        assert logits.grad[0, 2] == 0 and logits.grad[0, 0] != 0


def test_metrics():
    tags = ['a', 'b']
    category_tags = {'CurbRamp': ['a'], 'SurfaceProblem': ['b'], 'Obstacle': ['a'], 'Crosswalk': ['b']}
    masks = build_category_masks(tags, category_tags)
    categories = np.array([0, 0, 0, 2])
    labels = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=np.float32)
    probs = np.array([[0.9, 0.9], [0.1, 0.9], [0.4, 0.9], [0.8, 0.9]], dtype=np.float32)

    thresholds = tune_thresholds(probs, labels, categories, masks)
    assert 0.1 < thresholds[0, 0] <= 0.4
    assert thresholds[2, 0] == 0.5  # no positives

    predictions = apply_thresholds(probs, categories, thresholds, masks)
    assert not predictions[:, 1].any()  # b is invalid for these categories, even at 0.9

    results = per_tag_metrics(predictions, labels, categories, tags, category_tags)
    curb = results[results['category'] == 'CurbRamp'].iloc[0]
    assert (curb['tp'], curb['fp'], curb['fn']) == (2, 0, 0)
    assert len(results) == 4

    summary = summarize(results)
    assert summary['num_tags'] == 1 and summary['macro_f1'] == 1.0  # zero-support pairs are left out
    assert summarize(results, min_support=10)['num_tags'] == 0


def test_backbones():
    for backbone, size in (('cnn', 256), ('resnet50', 224), ('dinov2', 224), ('clip', 224)):
        model = SidewalkTagger(backbone, 6, size, pretrained=False).eval()
        with torch.no_grad():
            assert model(torch.zeros(1, 3, size, size)).shape == (1, 6)


def test_cnn_defaults(tmp, data_dir):
    argv = ['--data-dir', data_dir, '--output-dir', os.path.join(tmp, 'cnn'), '--num-workers', '0']
    args = train.parse_args(argv)
    assert args.backbone == 'cnn'
    assert (args.epochs, args.batch_size, args.image_size, args.lr, args.weight_decay) == (60, 64, 256, 0.01, 5e-4)

    path = train.main(argv + ['--epochs', '1', '--batch-size', '4'])
    assert os.path.basename(path) == 'cnn-bce.pth'
    model, checkpoint = load_checkpoint(path, torch.device('cpu'))
    assert checkpoint['image_size'] == 256 and isinstance(model.backbone, ConvNeuralNet)


def test_end_to_end(tmp, data_dir):
    common = ['--data-dir', data_dir, '--output-dir', os.path.join(tmp, 'checkpoints'), '--no-pretrained',
              '--backbone', 'resnet50', '--epochs', '1', '--batch-size', '4', '--image-size', '64',
              '--num-workers', '0']

    train.main(common + ['--loss', 'weighted_bce'])

    probe_path = train.main(common + ['--loss', 'focal', '--oversample', '--linear-probe'])
    assert os.path.basename(probe_path) == 'resnet50-focal-oversample-linear.pth'

    evaluate.main([probe_path, '--data-dir', data_dir, '--output-dir', os.path.join(tmp, 'results'),
                   '--batch-size', '4', '--num-workers', '0'])
    results = pd.read_csv(os.path.join(tmp, 'results', 'resnet50-focal-oversample-linear.csv'))
    assert len(results) == 8

    device = torch.device('cpu')
    model, checkpoint = load_checkpoint(probe_path, device)
    image = Image.new('RGB', (640, 640))
    predictions = predict_tags(model, checkpoint, image, 'Crosswalk', device)
    assert {tag for tag, _, _ in predictions} == {'bumpy', 'paint-fading'}


def test_linear_probe_freezes_backbone(tmp, data_dir):
    torch.manual_seed(0)
    args = train.parse_args(['--data-dir', data_dir, '--output-dir', os.path.join(tmp, 'probe'), '--no-pretrained',
                             '--backbone', 'resnet50', '--epochs', '1', '--batch-size', '4', '--image-size', '64', '--num-workers', '0',
                             '--linear-probe'])
    model = SidewalkTagger('resnet50', 6, 64, pretrained=False)
    model.freeze_backbone()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    checkpoint = torch.load(train.train_model(args))
    for name in ('backbone.bn1.running_mean', 'backbone.layer4.1.bn2.num_batches_tracked'):
        assert (checkpoint['state_dict'][name] == 0).all(), name


def main():
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, 'raw')
        data_dir = os.path.join(tmp, 'processed')
        write_raw_dataset(raw_dir)
        prepare_data.main(['--raw-dir', raw_dir, '--output-dir', data_dir])

        test_crop(tmp)
        jpeg_dir = test_zip_source(tmp, raw_dir)
        test_vocabulary(data_dir)
        test_losses_ignore_invalid_tags()
        test_metrics()
        test_backbones()
        test_cnn_defaults(tmp, data_dir)
        test_end_to_end(tmp, data_dir)
        test_linear_probe_freezes_backbone(tmp, data_dir)
        test_resume(tmp, jpeg_dir)
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
