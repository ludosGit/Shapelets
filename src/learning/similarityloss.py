from collections import OrderedDict
import warnings
from matplotlib.pyplot import axis

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import conv1d, pad
from tqdm import tqdm

class CorrelationSimilairty(nn.Module):
    """
    Calculates the max cross correlation of shapelets
    ----------
    """
    def __init__(self):
        super(CorrelationSimilairty, self).__init__()

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
# input shape is (n_series, n_channels, len_series)
# x = tensor([[1,1,1,1,5,5,5,5,1,1,1,1], [1,1,1,5,5,5,5,1,1,1,1,1]], dtype=torch.float).reshape(2,1,-1)
# x.shape
# padding = (6,6)
# x = pad(x, padding, mode="constant", value=0.0) 
# x.shape
# loss = ShapeletsSimilarityLoss()
# loss(x)
# y = tensor([[1,1,1,1,1,1,1,5,5,5,5,1], [1,1,1,1,1,1,1,5,5,5,5,1]]).reshape(2,1,-1)
# x = (x - torch.mean(x, dim=2, keepdims=True)) / torch.std(x, dim=2, keepdims=True)
# z = conv1d(x,y, padding=5)
# z.shape
# z = torch.max(z, dim=2)[0].float()
# torch.mean(z)


class DiscrepancySimilarity(nn.Module):
    """
    Calculates the max cross correlation of shapelets
    ----------
    """
    def __init__(self):
        super(DiscrepancySimilarity, self).__init__()

    def forward(self, s, sigma=1):
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
        padding = (round(L/2), round(L/2))
        s_padded = pad(s, padding, mode="constant", value=0.0) 
        patches = s_padded.unfold(dimension=1, size=n_channels, step=1).unfold(dimension=2, size=L, step=1)
        patches = patches.reshape(K, -1, n_channels, L)
        patches = torch.flatten(patches, start_dim=2, end_dim=3)
        shapelets = torch.flatten(s, start_dim=1, end_dim=2)
        output = torch.cdist(shapelets, patches, p=2)
        # hard min compared to soft-min from the paper
        discrepancy, _ = torch.min(output, dim=2) # has shape KxK
        discrepancy = torch.tril(discrepancy, diagonal=-1)
        combinations = K*(K-1)/2
        d = torch.sum(discrepancy)/combinations 
        return torch.exp(-torch.pow(d,2)/sigma)

#### TEST
# input shape is (n_series, n_channels, len_series)
# x = tensor([[1,1,1,1,5,5,5,5,1,1,1,1], [1,1,1,5,5,5,5,1,1,1,1,1]], dtype=torch.float).reshape(2,1,-1)
# x.shape
# # padding = (6,6)
# # x = pad(x, padding, mode="constant", value=0.0) 
# # x.shape
# loss = DiscrepancySimilairty()
# loss = loss(x)
# loss
# torch.exp(-torch.pow(torch.tensor(0.02),2)/1)

# y = tensor([[1,1,1,1,1,1,1,5,5,5,5,1], [1,1,1,1,1,1,1,5,5,5,5,1]]).reshape(2,1,-1)

# x = (x - torch.mean(x, dim=2, keepdims=True)) / torch.std(x, dim=2, keepdims=True)
# z = conv1d(x,y, padding=5)
# z.shape
# z = torch.max(z, dim=2)[0].float()
# torch.mean(z)