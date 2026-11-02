import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

TIMM_MODELS = {
    'dinov2': 'vit_base_patch14_reg4_dinov2.lvd142m',
    'clip': 'vit_base_patch16_clip_224.openai',
}


class ConvNeuralNet(nn.Module):
    """The network from classifier.py, up to fc1. Its fc2 and dropout are now in SidewalkTagger.
    Eight poolings take a 256x256 input down to 1x1, so it only accepts that size."""

    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 1 * 1, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(F.relu(self.bn8(self.conv8(x))))

        x = x.view(x.size(0), -1)
        return F.relu(self.fc1_bn(self.fc1(x)))


def build_backbone(name, image_size, pretrained):
    """Returns the backbone with its classifier removed, plus its feature size."""
    if name == 'cnn':
        return ConvNeuralNet(), 1024
    if name == 'resnet50':
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        num_features = net.fc.in_features
        net.fc = nn.Identity()
        return net, num_features
    net = timm.create_model(TIMM_MODELS[name], pretrained=pretrained, num_classes=0, img_size=image_size)
    return net, net.num_features


class SidewalkTagger(nn.Module):
    """Shared backbone with one sigmoid output per tag. Outputs logits; masking happens in the loss and at inference."""

    def __init__(self, backbone, num_tags, image_size=224, pretrained=True):
        super(SidewalkTagger, self).__init__()
        self.backbone, num_features = build_backbone(backbone, image_size, pretrained)
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(num_features, num_tags)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = SidewalkTagger(checkpoint['backbone'], len(checkpoint['tags']), checkpoint['image_size'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, checkpoint
