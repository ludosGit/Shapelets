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

    def forward(self, s):
        """
        Calculate the loss as the sum of the averaged cross correlation distances of the shapelets.
        Remind that the shapelets have shape (n_shapelets, n_channels, len_shapelets)
        @param shapelets: a list of the weights (as torch parameters) of the shapelet blocks
        @type shapelets: torch.parameter(tensor(float))
        @return: the computed loss
        @rtype: float
        """
        # return the max normalized cross correlation w.r.t. the different lags
        K, n_channels, L = s.shape
        s = (s - torch.mean(s, dim=2, keepdims=True)) / torch.std(s, dim=2, keepdims=True, unbiased=False)
        conv = conv1d(s, s, padding='same')/L
        m = torch.max(conv, axis=2).values
        m = torch.tril(m, diagonal=-1)
        combinations = K*(K-1)/2
        return torch.sum(m)/combinations


# #### TEST
# # input shape is (n_series, n_channels, len_series)
# x = tensor([[1,1,1,1,5,5,5,5,1,1,1,1], [1,1,1,5,5,5,5,1,1,1,1,1]], dtype=torch.float).reshape(2,1,-1)
# loss = ShapeletsSimilarityLoss()
# loss(x)
# y = tensor([[1,1,1,1,1,1,1,5,5,5,5,1], [1,1,1,1,1,1,1,5,5,5,5,1]]).reshape(2,1,-1)
# x = (x - torch.mean(x, dim=2, keepdims=True)) / torch.std(x, dim=2, keepdims=True)
# z = conv1d(x,y, padding=5)
# z.shape
# z = torch.max(z, dim=2)[0].float()
# torch.mean(z)
