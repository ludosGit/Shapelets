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
    """
    def __init__(self, dist_measure='euclidean'):
        super(L1DistanceLoss, self).__init__()

    def forward(self, x):
        """
        Calculate the loss as the average sum of the distances to each shapelet.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        x = x.clamp(1e-8)
        # avoid compiler warning
        # use L1 normalization
        y_loss = torch.mean(torch.sum(x, dim=1))
        return y_loss

# x = torch.Tensor([[1,2,3,4], [6,7,8,9], [5,10,11,4], [4,2,23,4], [6,2,81,9], [11,10,13,5]])
# # 6 rows for columns
# torch.sum(x, dim=1)
# # sum along the columns!!

class L2DistanceLoss(nn.Module):
    """
    Calculates the l2 similarity of a bunch of shapelets to a data set.
    """
    def __init__(self):
        super(L2DistanceLoss, self).__init__()

    def forward(self, x):
        """
        Calculate the loss as the average norm of the distances to each shapelet.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        x = x.clamp(1e-8)
        # avoid compiler warning
        y_loss = torch.mean(torch.norm(x, dim=1))
        return y_loss

# x = torch.Tensor([[1,2,3,4], [6,7,8,9], [5,10,11,4], [4,2,23,4], [6,2,81,9], [11,10,13,5]])
# # 6 rows for columns
# torch.norm(x, dim=1)
# # along the columns!!

class SVDD_L2DistanceLoss(nn.Module):
    """
    Calculates the l2 similarity of a bunch of shapelets to a data set.
    """
    def __init__(self, radius):
        super(SVDD_L2DistanceLoss, self).__init__()
        self.radius = radius

    def update_r(self, radius):
        self.radius = radius
        return None

    def get_radius(self):
        return np.sqrt(self.radius)

    def forward(self, x):
        """
        Calculate the loss as the average norm of the distances to each shapelet.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        x = x.clamp(1e-8)
        # avoid compiler warning
        l1 = torch.norm(x, dim=1)
        # the radius is squared!!
        l2 = l1 - self.radius

        l2[l2 < 0] = 0 
        l2 = torch.mean(l2)
        loss = l2 + torch.mean(l1)
        return loss

# x = torch.Tensor([[1,2,3,4], [6,7,8,9], [5,10,11,4], [4,2,23,4], [6,2,81,9], [11,10,13,5]])
# # 6 rows for columns
# torch.norm(x, dim=1) - 1
# # along the columns!!

# x = torch.Tensor([1,2,0,0])
# torch.count_nonzero(x)

