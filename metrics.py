import numpy as np
import pandas as pd
import torch

from dataset import CATEGORIES

# Liu et al. only report tags with at least this many positives in the test set.
# On the validated test set this leaves the 33 category/tag pairs in the paper.
BENCHMARK_MIN_SUPPORT = 10


def collect_predictions(model, dataloader, device):
    model.eval()
    all_probs, all_labels, all_categories = [], [], []
    with torch.no_grad():
        for images, labels, categories in dataloader:
            logits = model(images.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
            all_categories.append(categories.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_categories)


def f1_score(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def tune_thresholds(probs, labels, categories, category_masks):
    """Pick the F1-maximising threshold per (category, tag) pair. Pairs with no positives keep 0.5."""
    thresholds = np.full(category_masks.shape, 0.5)
    candidates = np.arange(0.05, 0.96, 0.05)
    for c in range(len(CATEGORIES)):
        rows = categories == c
        for t in np.flatnonzero(category_masks[c]):
            y_true = labels[rows, t] > 0.5
            if not y_true.any():
                continue
            scores = []
            for candidate in candidates:
                y_pred = probs[rows, t] >= candidate
                scores.append(f1_score((y_pred & y_true).sum(), (y_pred & ~y_true).sum(), (~y_pred & y_true).sum()))
            thresholds[c, t] = candidates[int(np.argmax(scores))]
    return thresholds


def apply_thresholds(probs, categories, thresholds, category_masks):
    """Invalid tags are forced to zero."""
    return (probs >= thresholds[categories]) & category_masks[categories]


def per_tag_metrics(predictions, labels, categories, tags, category_tags):
    """One row per (category, tag) pair, the unit Liu et al. report, so a shared tag is scored per category."""
    rows = []
    for c, category in enumerate(CATEGORIES):
        in_category = categories == c
        for tag in category_tags[category]:
            t = tags.index(tag)
            y_true = labels[in_category, t] > 0.5
            y_pred = predictions[in_category, t]
            tp = int((y_pred & y_true).sum())
            fp = int((y_pred & ~y_true).sum())
            fn = int((~y_pred & y_true).sum())
            rows.append({
                'category': category, 'tag': tag, 'support': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn,
                'precision': tp / (tp + fp) if tp + fp else 0.0,
                'recall': tp / (tp + fn) if tp + fn else 0.0,
                'f1': f1_score(tp, fp, fn),
            })
    return pd.DataFrame(rows)


def summarize(results, min_support=1):
    """Tags below min_support are left out so a tag with no positives can't drag macro F1 down."""
    kept = results[results['support'] >= min_support]
    if kept.empty:
        return {'num_tags': 0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'weighted_f1': 0.0}
    return {
        'num_tags': len(kept),
        'macro_f1': kept['f1'].mean(),
        'micro_f1': f1_score(kept['tp'].sum(), kept['fp'].sum(), kept['fn'].sum()),
        'weighted_f1': np.average(kept['f1'], weights=kept['support']),
    }
