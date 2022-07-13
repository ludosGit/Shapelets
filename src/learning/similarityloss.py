from collections import OrderedDict
import warnings
from matplotlib.pyplot import axis

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import conv1d
from tqdm import tqdm

class ShapeletsSimilarityLoss(nn.Module):
    """
    Calculates the max cross correlation of shapelets
    ----------
    """
    def __init__(self):
        super(ShapeletsSimilarityLoss, self).__init__()

    def cross_corr_distance(self, x1, x2=None):
        """
        Calculate the max cross correlation similarity between all pairs of x1 and x2.
        x2 can be left zero, in case the similarity between solely all pairs in x1 shall be computed.
        Remind that the shapelets have shape (n_shapelets, n_channels, len_shapelets)
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

        L = x1.shape[2]
        # TODO: check all the shapes
        # first normalize
        # x1 = (x1 - torch.mean(x1, dim=1)) / torch.std(x1, dim=1)
        # x2 = (x1 - torch.mean(x2, dim=1)) / torch.std(x2, dim=1)
        # divide by the length
        z = conv1d(x1, x2, padding=round(L/2)) / L
        z = torch.max(z, dim=2)[0].float()
        # return the max normalized cross correlation w.r.t. the different lags
        return z

    def forward(self, shapelets):
        """
        Calculate the loss as the sum of the averaged cross correlation distances of the shapelets.
        @param shapelets: a list of the weights (as torch parameters) of the shapelet blocks
        @type shapelets: torch.parameter(tensor(float))
        @return: the computed loss
        @rtype: float
        """
        return torch.mean(self.cross_corr_distance(shapelets))


# # input shape is (n_series, n_channels, len_series)
# x = tensor([[1,1,1,1,5,5,5,5,1,1,1,1], [1,1,1,1,5,5,5,5,1,1,1,1]]).reshape(2,1,-1)
# y = tensor([[1,1,1,1,1,1,1,5,5,5,5,1], [1,1,1,1,1,1,1,5,5,5,5,1]]).reshape(2,1,-1)
# x.shape
# z = conv1d(x,y, padding=5)
# z.shape
# z = torch.max(z, dim=2)[0].float()
# torch.mean(z)
