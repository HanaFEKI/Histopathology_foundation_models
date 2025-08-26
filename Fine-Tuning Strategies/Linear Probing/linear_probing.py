import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    """
    Linear probing head on top of a frozen backbone.
    """
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Linear classifier for probing
        self.probe = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.probe(features)
        return logits

# Example usage
if __name__ == "__main__":
    # Dummy backbone returning feature vectors
    class DummyBackbone(nn.Module):
        def forward(self, x):
            return torch.mean(x, dim=(2, 3))  # e.g., global average pooling

    x = torch.randn(8, 3, 224, 224)  # batch of 8, 3x224x224 images
    backbone = DummyBackbone()
    probe = LinearProbe(backbone, embed_dim=3, num_classes=10)
    y = probe(x)
    print(y.shape)  # [8, 10]
