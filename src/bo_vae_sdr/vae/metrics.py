from typing import Optional, Union

import numpy as np
import torch
from pytorch_metric_learning import distances
from torch import Tensor

class TripletLossTorch:
    def __init__(
        self,
        eta: float,
        margin: Optional[float] = None,
        soft: Optional[bool] = False,
        nu: Optional[float] = None,
    ):
        """
        Compute Triplet loss
        Args:
            eta: separate positive and negative elements in temrs of `y` distance
            margin: hard triplet loss parameter
            soft: whether to use sigmoid version of triplet loss
            nu: parameter of hyperbolic function softening transition between positive and negative classes
        """
        self.eta = eta
        self.margin = margin
        self.soft = soft
        assert nu is None or nu > 0, nu
        self.nu = nu

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        lpembdist = distances.LpDistance(
            normalize_embeddings=False, p=2, power=1)
        emb_distance_matrix = lpembdist(embs)

        lpydist = distances.LpDistance(
            normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        positive_embs = emb_distance_matrix.where(
            y_distance_matrix <= self.eta, torch.tensor(0.0).to(embs)
        )
        negative_embs = emb_distance_matrix.where(
            y_distance_matrix > self.eta, torch.tensor(0.0).to(embs)
        )

        loss_loop = 0 * torch.tensor([0.0], requires_grad=True).to(embs)
        n_positive_triplets = 0
        for i in range(embs.size(0)):
            pos_i = positive_embs[i][positive_embs[i] > 0]
            neg_i = negative_embs[i][negative_embs[i] > 0]
            pairs = torch.cartesian_prod(pos_i, -neg_i)
            if self.soft:
                triplet_losses_for_anchor_i = torch.nn.Softplus()(
                    pairs.sum(dim=-1)
                )
                if self.nu is not None:
                    # get the corresponding delta ys
                    pos_y_i = y_distance_matrix[i][positive_embs[i] > 0]
                    neg_y_i = y_distance_matrix[i][negative_embs[i] > 0]
                    pairs_y = torch.cartesian_prod(pos_y_i, neg_y_i)
                    assert pairs.shape == pairs_y.shape, (
                        pairs_y.shape, pairs.shape)
                    ####################################################################
                    # This is exactly the L_{s-trip} in the thesis
                    triplet_losses_for_anchor_i = (
                        triplet_losses_for_anchor_i
                        * self.smooth_indicator(self.eta - pairs_y[:, 0]).div(
                            self.smooth_indicator(self.eta)
                        )  # positive pairs w_{ij} or w^p###
                        * self.smooth_indicator(pairs_y[:, 1] - self.eta).div(
                            self.smooth_indicator(1 - self.eta)
                        )  # negative pairs w_{ik} or w^n
                    ) * self.identity_indicator(
                        pos_ys=pairs_y[:, 0], neg_ys=pairs_y[:, 1]
                    )
            else:
                triplet_losses_for_anchor_i = torch.relu(
                    self.margin + pairs.sum(dim=-1)
                )
            n_positive_triplets += (triplet_losses_for_anchor_i > 0).sum()
            loss_loop += triplet_losses_for_anchor_i.sum()
        loss_loop = loss_loop.div(max(1, n_positive_triplets))

        return loss_loop

    def identity_indicator(self, pos_ys, neg_ys):
        return torch.where(
            torch.logical_and(pos_ys < self.eta , neg_ys >= self.eta),
            1,
            0        
        )
    
    def smooth_indicator(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, float):
            return np.tanh(x / (2 * self.nu))
        return torch.tanh(x / (2 * self.nu))

    def compute_loss(self, embs: Tensor, ys: Tensor):
        return self.build_loss_matrix(embs, ys)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)
