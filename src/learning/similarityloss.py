from collections import OrderedDict
import warnings

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class ShapeletsSimilarityLoss(nn.Module):
    """
    Calculates the max cross correlation of shapelets
    ----------
    """
    def __init__(self):
        super(ShapeletsSimilarityLoss, self).__init__()

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        """
        Calculate the cosine similarity between all pairs of x1 and x2. x2 can be left zero, in case the similarity
        between solely all pairs in x1 shall be computed.
        @param x1: the first set of input vectors
        @type x1: tensor(float)
        @param x2: the second set of input vectors
        @type x2: tensor(float)
        @param eps: add small value to avoid division by zero.
        @type eps: float
        @return: a distance matrix containing the cosine similarities
        @type: tensor(float)
        """
        x2 = x1 if x2 is None else x2
        # unfold time series to emulate sliding window
        x1 = x1.unfold(2, x2.shape[2], 1).contiguous()
        x1 = x1.transpose(0, 1)
        # normalize with l2 norm
        x1 = x1 / x1.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        x2 = x2 / x2.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)

        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x1 = torch.matmul(x1, x2.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        # and average over dims to keep range between 0 and 1
        n_dims = x1.shape[1]
        x1 = torch.sum(x1, dim=1) / n_dims
        return x1

    def forward(self, shapelet_blocks):
        """
        Calculate the loss as the sum of the averaged cosine similarity of the shapelets in between each block.
        @param shapelet_blocks: a list of the weights (as torch parameters) of the shapelet blocks
        @type shapelet_blocks: list of torch.parameter(tensor(float))
        @return: the computed loss
        @rtype: float
        """
        losses = 0.
        for block in shapelet_blocks:
            shapelets = block[1]
            shapelets.retain_grad()
            sim = self.cosine_distance(shapelets, shapelets)
            losses += torch.mean(sim)
        return losses
