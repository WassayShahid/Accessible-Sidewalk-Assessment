import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (SidewalkDataset, build_category_masks, build_transform, build_vocabulary, compute_pos_weight,
                     load_samples, rare_tag_sampler, train_val_split)
from losses import MaskedBCELoss, MaskedFocalLoss
from metrics import apply_thresholds, collect_predictions, per_tag_metrics, summarize, tune_thresholds
from model import SidewalkTagger


# The cnn values are the ones from classifier.py.
CNN_DEFAULTS = {'epochs': 60, 'batch_size': 64, 'image_size': 256, 'lr': 0.01, 'head_lr': 0.01, 'weight_decay': 5e-4}
PRETRAINED_DEFAULTS = {'epochs': 15, 'batch_size': 32, 'image_size': 224, 'lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': 1e-4}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train the category-masked sidewalk tagger')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--backbone', choices=['cnn', 'resnet50', 'dinov2', 'clip'], default='cnn')
    parser.add_argument('--loss', choices=['bce', 'weighted_bce', 'focal'], default='bce')
    parser.add_argument('--oversample', action='store_true', help='oversample images with rare tags')
    parser.add_argument('--linear-probe', action='store_true', help='freeze the backbone and train only the head')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--image-size', type=int, help='cnn needs 256; dinov2 needs a multiple of 14')
    parser.add_argument('--lr', type=float, help='backbone learning rate')
    parser.add_argument('--head-lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    defaults = CNN_DEFAULTS if args.backbone == 'cnn' else PRETRAINED_DEFAULTS
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if args.backbone == 'cnn' and args.image_size != 256:
        parser.error('--backbone cnn needs --image-size 256')
    if args.backbone == 'cnn' and args.linear_probe:
        parser.error('--linear-probe needs a pretrained backbone; cnn starts from random weights')
    return args


def run_name(args):
    parts = [args.backbone, args.loss]
    if args.oversample:
        parts.append('oversample')
    if args.linear_probe:
        parts.append('linear')
    return '-'.join(parts)


def build_criterion(args, train_samples, category_masks):
    if args.loss == 'focal':
        return MaskedFocalLoss()
    pos_weight = compute_pos_weight(train_samples, category_masks) if args.loss == 'weighted_bce' else None
    return MaskedBCELoss(pos_weight)


def validate(model, val_loader, device, tags, category_tags, category_masks):
    probs, labels, categories = collect_predictions(model, val_loader, device)
    thresholds = tune_thresholds(probs, labels, categories, category_masks)
    predictions = apply_thresholds(probs, categories, thresholds, category_masks)
    results = per_tag_metrics(predictions, labels, categories, tags, category_tags)
    return summarize(results), thresholds


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tags, category_tags = build_vocabulary(args.data_dir)
    category_masks = build_category_masks(tags, category_tags)
    masks_tensor = torch.from_numpy(category_masks).float().to(device)
    print(f'{len(tags)} tags in the head')

    samples = load_samples(args.data_dir, 'train', tags, category_tags)
    train_samples, val_samples = train_val_split(samples, seed=args.seed)
    print(f'train: {len(train_samples)}, val: {len(val_samples)}')

    train_set = SidewalkDataset(train_samples, build_transform(args.backbone, args.image_size, train=True))
    val_set = SidewalkDataset(val_samples, build_transform(args.backbone, args.image_size))

    sampler = rare_tag_sampler(train_samples) if args.oversample else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)

    model = SidewalkTagger(args.backbone, len(tags), args.image_size, args.pretrained).to(device)
    criterion = build_criterion(args, train_samples, category_masks).to(device)

    if args.linear_probe:
        model.freeze_backbone()
    param_groups = [{'params': model.head.parameters(), 'lr': args.head_lr}]
    if not args.linear_probe:
        param_groups.append({'params': model.backbone.parameters(), 'lr': args.lr})
    if args.backbone == 'cnn':
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f'{run_name(args)}.pth')
    best_macro_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        if args.linear_probe:
            model.backbone.eval()  # keeps BatchNorm statistics frozen too
        running_loss = 0.0

        for inputs, labels, categories in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            masks = masks_tensor[categories.to(device)]
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        summary, thresholds = validate(model, val_loader, device, tags, category_tags, category_masks)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Val macro F1: {summary['macro_f1']:.4f}, Val micro F1: {summary['micro_f1']:.4f}")

        if summary['macro_f1'] > best_macro_f1:
            best_macro_f1 = summary['macro_f1']
            torch.save({
                'state_dict': model.state_dict(),
                'backbone': args.backbone,
                'image_size': args.image_size,
                'tags': tags,
                'category_tags': category_tags,
                'thresholds': thresholds.tolist(),
                'epoch': epoch + 1,
                'val_macro_f1': float(best_macro_f1),
            }, checkpoint_path)

    print(f'Best val macro F1: {best_macro_f1:.4f}, saved to {checkpoint_path}')
    return checkpoint_path


def main(argv=None):
    return train_model(parse_args(argv))


if __name__ == '__main__':
    main()
