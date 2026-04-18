import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, logger=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if logger:
            self.logger = logger

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(dim=1).clamp(min=1e-8).sqrt()

    def calc_cosine(self, x1, x2):
        dot_product_sum = (x1 * x2).sum(dim=1)
        norm_multiply = (x1.pow(2).sum(dim=1).sqrt()) * (x2.pow(2).sum(dim=1).sqrt())
        return dot_product_sum / norm_multiply
    
    def calc_cosine_distance(self, x1, x2):
        # PyTorch built-in is highly optimized and handles zero-division safely
        # Cosine distance ranges from 0.0 (identical) to 2.0 (complete opposites)
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def calc_manhattan(self, x1, x2):
        return (x1 - x2).abs().sum(dim=1)

    def forward(self, anchor, positive, negative):
        # 1. Calculate Cosine Distances
        distance_positive = self.calc_cosine_distance(anchor, positive)
        distance_negative = self.calc_cosine_distance(anchor, negative)
        
        # 2. Apply Standard Triplet Loss Formula
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        # 3. NaN checking
        if (
            not (anchor.isnan().any())
            and not (positive.isnan().any())
            and not (negative.isnan().any())
        ):
            if (losses.isnan().any()) and hasattr(self, 'logger'):
                self.logger.info("losses has NaN")

        return losses.mean()


class PrototypeStructuralLoss(nn.Module):
    """
    Calculates Prototype Diversity Loss (PDL), R1, and R2 structural losses on the Cosine Hypersphere
    using Normalized Squared Euclidean distances.
    Gee, A. H., Garcia-Olano, D., Ghosh, J., & Paydarfar, D. (2019, April 18). Explaining Deep Classification of Time-Series Data with Learned Prototypes. arXiv.org. https://arxiv.org/abs/1904.08935
    """

    def __init__(self, eps=1e-4):
        super(PrototypeStructuralLoss, self).__init__()
        self.eps = eps

    def forward(self, latents, prototypes):
        # 1. Project to the Unit Hypersphere
        z_norm = F.normalize(latents, p=2, dim=1)
        p_norm = F.normalize(prototypes, p=2, dim=1)

        m = p_norm.shape[0]
        n = z_norm.shape[0]

        # 2. R1 & R2 Losses (Data-to-Prototype)
        # Dimension (num_latents, num_prototypes) matrix of squared Euclidean distances
        # ||z - p||^2 = 2 - 2*(z @ p.T)
        dist_data_proto = 2.0 - 2.0 * torch.matmul(z_norm, p_norm.T)
        # Clamp to avoid tiny negative numbers due to float32 precision errors
        dist_data_proto = dist_data_proto.clamp(min=0.0)
        # Minimising R1 loss term promotes each prototype vector to learn one of the encoded training example
        r1_loss = torch.mean(torch.min(dist_data_proto, dim=0)[0])
        # Minimising R2 loss term promotes each encoded training example to be close to at least one prototype vector
        r2_loss = torch.mean(torch.min(dist_data_proto, dim=1)[0])

        # 3. PDL Loss (Prototype-to-Prototype)
        dist_proto_proto = 2.0 - 2.0 * torch.matmul(p_norm, p_norm.T)
        dist_proto_proto = dist_proto_proto.clamp(min=0.0)
        mask = torch.eye(m, device=p_norm.device).bool()
        dist_proto_proto.masked_fill_(mask, float("inf"))

        min_distances, _ = torch.min(dist_proto_proto, dim=1)
        avg_min_dist = torch.mean(min_distances)

        # Inverse log penalty
        pdl_loss = 1.0 / (torch.log(avg_min_dist + 1.0) + self.eps)

        return r1_loss, r2_loss, pdl_loss
