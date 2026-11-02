import timm
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

TIMM_MODELS = {
    'dinov2': 'vit_base_patch14_reg4_dinov2.lvd142m',
    'clip': 'vit_base_patch16_clip_224.openai',
}


def build_backbone(name, image_size, pretrained):
    """Returns the backbone with its classifier removed, plus its feature size."""
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
