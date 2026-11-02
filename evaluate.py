import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CATEGORIES, SidewalkDataset, build_category_masks, build_transform, load_samples
from metrics import BENCHMARK_MIN_SUPPORT, apply_thresholds, collect_predictions, per_tag_metrics, summarize
from model import load_checkpoint


def print_summary(name, summary):
    print(f"{name:<34} tags={summary['num_tags']:<3} macro F1={summary['macro_f1']:.4f}  "
          f"micro F1={summary['micro_f1']:.4f}  weighted F1={summary['weighted_f1']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint on the test set')
    parser.add_argument('checkpoint')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    tags, category_tags = checkpoint['tags'], checkpoint['category_tags']
    category_masks = build_category_masks(tags, category_tags)
    thresholds = np.array(checkpoint['thresholds'])

    samples = load_samples(args.data_dir, 'test', tags, category_tags)
    test_set = SidewalkDataset(samples, build_transform(checkpoint['backbone'], checkpoint['image_size']))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    probs, labels, categories = collect_predictions(model, test_loader, device)
    predictions = apply_thresholds(probs, categories, thresholds, category_masks)
    results = per_tag_metrics(predictions, labels, categories, tags, category_tags)

    print(results.sort_values(['category', 'support'], ascending=[True, False]).to_string(index=False, float_format='%.4f'))
    print()
    print_summary(f'Benchmark (support >= {BENCHMARK_MIN_SUPPORT})', summarize(results, BENCHMARK_MIN_SUPPORT))
    print_summary('All tags with a test positive', summarize(results))
    for category in CATEGORIES:
        print_summary(f'{category} (benchmark tags)',
                      summarize(results[results['category'] == category], BENCHMARK_MIN_SUPPORT))

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    output_path = os.path.join(args.output_dir, f'{name}.csv')
    results.to_csv(output_path, index=False)
    print(f'\nPer-tag results saved to {output_path}')


if __name__ == '__main__':
    main()
