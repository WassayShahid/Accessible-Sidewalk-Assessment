import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import CATEGORIES, build_category_masks, build_transform
from model import load_checkpoint


def predict_tags(model, checkpoint, image, category, device):
    """Returns (tag, probability, passed threshold) for the tags valid for this category, most likely first."""
    tags = checkpoint['tags']
    c = CATEGORIES.index(category)
    category_masks = build_category_masks(tags, checkpoint['category_tags'])
    thresholds = np.array(checkpoint['thresholds'])

    preprocess = build_transform(checkpoint['backbone'], checkpoint['image_size'])
    image_tensor = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(image_tensor))[0].cpu().numpy()

    predictions = [(tags[t], float(probabilities[t]), bool(probabilities[t] >= thresholds[c, t]))
                   for t in np.flatnonzero(category_masks[c])]
    return sorted(predictions, key=lambda item: item[1], reverse=True)


def pick_image():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title='Select an Image', filetypes=[('Image files', '*.png *.jpg *.jpeg')])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Tag one sidewalk crop')
    parser.add_argument('checkpoint')
    parser.add_argument('--category', choices=CATEGORIES, required=True)
    parser.add_argument('--image', help='image path; opens a file picker if omitted')
    args = parser.parse_args(argv)

    file_path = args.image or pick_image()
    if not file_path:
        print('No image selected.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    image = Image.open(file_path)
    predictions = predict_tags(model, checkpoint, image, args.category, device)

    detected = [tag for tag, _, passed in predictions if passed]
    print(f'Detected: {", ".join(detected) if detected else "no tags"}\n')
    for tag, probability, passed in predictions:
        print(f"{'*' if passed else ' '} {tag:28s} {probability*100:.2f}%")

    plt.imshow(image)
    plt.axis('off')
    plt.title(f"{args.category}: {', '.join(detected) if detected else 'no tags'}")
    plt.show()


if __name__ == '__main__':
    main()
