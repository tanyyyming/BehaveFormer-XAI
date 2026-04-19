import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, logger=None):
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
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        if (
            not (anchor.isnan().any())
            and not (positive.isnan().any())
            and not (negative.isnan().any())
        ):
            if (losses.isnan().any()) and self.logger:
                self.logger.info("losses has NaN")

        return losses.mean()


class PrototypeStructuralLoss(nn.Module):
    """
    Calculates Prototype Diversity Loss (PDL), R1, and R2 structural losses 
    using Unconstrained Euclidean Distance.
    Gee, A. H., Garcia-Olano, D., Ghosh, J., & Paydarfar, D. (2019, April 18). Explaining Deep Classification of Time-Series Data with Learned Prototypes. arXiv.org. https://arxiv.org/abs/1904.08935
    """

    def __init__(self, eps=1e-4):
        super(PrototypeStructuralLoss, self).__init__()
        self.eps = eps

    def forward(self, latents, prototypes):
        # 1. NO NORMALIZATION: Operate in unconstrained Euclidean space
        m = prototypes.shape[0]
        n = latents.shape[0]

        # 2. R1 & R2 Losses (Data-to-Prototype)
        # Calculate pairwise squared Euclidean distances: ||z - p||^2 = ||z||^2 + ||p||^2 - 2(z @ p.T)
        z_sq = latents.pow(2).sum(dim=1, keepdim=True)        # Shape: (n, 1)
        p_sq = prototypes.pow(2).sum(dim=1).unsqueeze(0)      # Shape: (1, m)
        
        dist_sq_data_proto = z_sq + p_sq - 2.0 * torch.matmul(latents, prototypes.T)
        
        # Apply clamp BEFORE sqrt to avoid NaN gradients when distance is exactly 0
        dist_data_proto = dist_sq_data_proto.clamp(min=1e-8).sqrt()

        # R1: Every prototype must be close to at least one training example
        r1_loss = torch.mean(torch.min(dist_data_proto, dim=0)[0])
        
        # R2: Every training example must be close to at least one prototype
        r2_loss = torch.mean(torch.min(dist_data_proto, dim=1)[0])

        # 3. PDL Loss (Prototype-to-Prototype)
        p_sq_col = prototypes.pow(2).sum(dim=1, keepdim=True) # Shape: (m, 1)
        p_sq_row = prototypes.pow(2).sum(dim=1).unsqueeze(0)  # Shape: (1, m)
        
        dist_sq_proto_proto = p_sq_col + p_sq_row - 2.0 * torch.matmul(prototypes, prototypes.T)
        dist_proto_proto = dist_sq_proto_proto.clamp(min=1e-8).sqrt()
        
        # Mask the diagonal (distance to itself) with infinity so it isn't picked as the minimum
        mask = torch.eye(m, device=prototypes.device).bool()
        dist_proto_proto = dist_proto_proto.masked_fill(mask, float("inf"))

        min_distances, _ = torch.min(dist_proto_proto, dim=1)
        avg_min_dist = torch.mean(min_distances)

        # Inverse log penalty for PDL
        pdl_loss = 1.0 / (torch.log(avg_min_dist + 1.0) + self.eps)

        return r1_loss, r2_loss, pdl_loss
