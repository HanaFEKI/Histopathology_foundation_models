import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        MoCo Model.
        Args:
            base_encoder: backbone model (e.g., ResNet)
            dim: feature dimension (output of projection head)
            K: queue size
            m: momentum for updating key encoder
            T: softmax temperature
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # Create the encoders : the query encoder f_q and the key encoder f_k must have the same architecture 
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # Initialize encoder_k weights to encoder_q
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # stop gradient for key encoder

    # Create the queue of negative keys (feature dim x queue size), normalized
    self.register_buffer("queue", torch.randn(dim, K))  # Initialized with random features
    self.queue = F.normalize(self.queue, dim=0)  # Normalize each column (key) to unit length
    
    # Pointer to track the position for next key insertion in the queue
    self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the queue with the latest keys.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images (augmented differently from im_q)
        Output:
            loss
        """
        # Compute query features
        q = self.encoder_q(im_q)  # NxC
        q = F.normalize(q, dim=1)

        # Compute key features with no gradient
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)  # NxC
            k = F.normalize(k, dim=1)

        # Positive logits: Nx1 : similarity scores between q (queries) and k (positive keys, same instance)
        # [N, C] (batch of N samples, feature dim C)
        # einsum stands for Einstein summation convention.
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #  output shape [N] -> unsqueeze(-1) -> makes it [N, 1] so it can be concatenated later.

        # Negative logits: NxK
        # q: [N, C], queue: [C, K]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits: Nx(1+K) : Now, for each query, the first column is the positive similarity, and the rest are negatives.
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.T

        # Labels: positive key is the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Contrastive loss
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        return loss
