import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, encoder_f, projector_g, t1, t2, temperature):
        super(SimCLR, self).__init__()
        self.f = encoder_f    # e.g., ResNet without FC
        self.g = projector_g  # e.g., MLP
        self.t1 = t1          # transform 1
        self.t2 = t2          # transform 2
        self.tau = temperature

    def augment_and_project(self, x):
        x1 = self.t1(x)
        x2 = self.t2(x)
        z1 = self.g(self.f(x1))
        z2 = self.g(self.f(x2))
        return z1, z2

    def forward(self, x):
        batch_size = x.size(0)
        z1, z2 = self.augment_and_project(x)

        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)  # shape: (2N, D)
        similarity_matrix = torch.matmul(representations, representations.T)  # (2N, 2N)

        # Remove self-similarity
        mask = ~torch.eye(2 * batch_size, device=x.device).bool()
        similarity_matrix = similarity_matrix[mask].view(2 * batch_size, -1)

        positives = torch.sum(z1 * z2, dim=1)  # shape: (N,)
        positives = torch.cat([positives, positives], dim=0)  # shape: (2N,)

        logits = similarity_matrix / self.tau
        labels = torch.arange(2 * batch_size, device=x.device)
        labels = (labels + batch_size) % (2 * batch_size)

        loss = F.cross_entropy(logits, labels)
        return loss
