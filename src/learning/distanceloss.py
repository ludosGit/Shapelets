from collections import OrderedDict
import warnings

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class L1DistanceLoss(nn.Module):
    """
    Calculates the l1 similarity of a bunch of shapelets to a data set.
    It is the one we used as heuristic in the search methods.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, dist_measure='euclidean'):
        super(L1DistanceLoss, self).__init__()
        if not dist_measure == 'euclidean' and not dist_measure == 'cosine':
            raise ValueError("Parameter 'dist_measure' must be either of 'euclidean' or 'cosine'.")
        self.dist_measure = dist_measure

    def forward(self, x):
        """
        Calculate the loss as the average sum of the distances to each shapelet.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        # torch.topk: Returns the k largest (or smallest if largest=False) elements of the given input tensor along a given dimension.
        x = x.clamp(1e-8)
        # avoid compiler warning
        y_loss = None
        ## use L1 normalization
        if self.dist_measure == 'euclidean':
            y_loss = torch.mean(torch.sum(x, dim=1))
        return y_loss

# x = torch.Tensor([[1,2,3,4], [6,7,8,9], [5,10,11,4], [4,2,23,4], [6,2,81,9], [11,10,13,5]])
# # 6 rows for columns
# torch.sum(x, dim=1)
# # sum along the columns!!

class L2DistanceLoss(nn.Module):
    """
    Calculates the l2 similarity of a bunch of shapelets to a data set.
    It is the one we used as heuristic in the search methods.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, dist_measure='euclidean'):
        super(L2DistanceLoss, self).__init__()
        if not dist_measure == 'euclidean' and not dist_measure == 'cosine':
            raise ValueError("Parameter 'dist_measure' must be either of 'euclidean' or 'cosine'.")
        self.dist_measure = dist_measure

    def forward(self, x):
        """
        Calculate the loss as the average sum of the distances to each shapelet.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        # torch.topk: Returns the k largest (or smallest if largest=False) elements of the given input tensor along a given dimension.
        x = x.clamp(1e-8)
        # avoid compiler warning
        y_loss = None
        ## use L1 normalization
        if self.dist_measure == 'euclidean':
            y_loss = torch.mean(torch.norm(x, dim=1))
        return y_loss

x = torch.Tensor([[1,2,3,4], [6,7,8,9], [5,10,11,4], [4,2,23,4], [6,2,81,9], [11,10,13,5]])
# 6 rows for columns
torch.norm(x, dim=1)
# along the columns!!
