from collections import OrderedDict
import warnings

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


### EUCLIDEAN DISTANCE LAYER:
class DistanceLayer(nn.Module):
    """
    Calculates the euclidean distances of a bunch of shapelets to a data using a sliding window and performs global min-pooling.
    Parameters
    ----------
    len_shapelets : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, len_shapelets, num_shapelets, in_channels=1, to_cuda=True):
        super(DistanceLayer, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.len_shapelets = len_shapelets
        self.in_channels = in_channels

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.num_shapelets, self.in_channels, self.len_shapelets, requires_grad=True)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets)
        
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x, mean_center=True):
        """
        1) Unfold the data set with sliding window 2) flatten() the patches w.r.t. the channel dimension 
        3) calculate euclidean distance and 4) perform global min-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the euclidean for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        # unfold time series to emulate sliding window
        
        num_samples, _, _ = x.shape
        patches = x.unfold(dimension=1, size=self.in_channels, step=1).unfold(dimension=2, size=self.len_shapelets, step=1)
        patches = patches.reshape(num_samples, -1, self.in_channels, self.len_shapelets)
        if mean_center:
            ####
            # added mean shift:
            patches = patches - torch.mean(patches, dim=3, keepdim=True)
            ####
        patches = torch.flatten(patches, start_dim=2, end_dim=3)
        shapelets = torch.flatten(self.shapelets, start_dim=1, end_dim=2)

        output = torch.cdist(shapelets, patches)

        # hard min compared to soft-min from the paper
        output_final, _ = torch.min(output, dim=2)
        if self.num_shapelets == 1:
            output_final = output_final.reshape((-1,1))
        return output_final

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)}"
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)
        self.shapelets.retain_grad()

    # not used
    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)}"
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[j,:,:] = weights
        self.shapelets = nn.Parameter(self.shapelets)
        self.shapelets.retain_grad()



# ################ TEST 

# # try with 3 channels
# X = [[[1,2,3,4], [6,7,8,9], [5,10,11,4]], [[4,2,23,4], [6,2,81,9], [11,10,13,5]]]
# # NOTE: to forward x must be a tensor(float) of shape (num_samples, in_channels, len_ts)
# X = torch.Tensor(X)
# print('shape of input dataset', X.shape) 
# num_samples, in_channels, len_ts = X.shape

# shapelets = np.array([[[1,1],[1,1], [1,1]], [[0,1], [0,1],[0,1]]])
# print('shapelets of dim', shapelets.shape) # (num_shapelets, in_channels, shapelets_size)

# n_shapelets, _, len_shapelets = shapelets.shape 
# ## set shapelets as parameters of the layer
# layer = DistanceLayer(shapelets_size=len_shapelets, num_shapelets=n_shapelets, in_channels=in_channels, to_cuda=False)
# layer.set_shapelet_weights(shapelets)
# shapelets = layer.get_shapelets()
# shapelets # same with added gradient 
# shapelets.shape
# # OK
# X
# patches = X.unfold(dimension=1, size=in_channels, step=1).unfold(dimension=2, size=len_shapelets, step=1)
# patches = patches.reshape(num_samples, -1, in_channels, len_shapelets)
# patches[0] 
# patches.shape

# # flatten the multivariate patches of time series to compute euclidean distance
# patches = torch.flatten(patches, start_dim=2, end_dim=3)
# patches
# patches.shape
# pdist = nn.PairwiseDistance(p=2)

# shapelets = torch.flatten(shapelets, start_dim=1, end_dim=2)
# # gradient is taken into consideration!!
# shapelets

# patches[1][0] - shapelets[1]
# output = torch.cdist(shapelets, patches)

# # test if it is the same:
# for i in range(len(patches)):
#     print(pdist(shapelets, patches[i]))


# # torch.min Returns a namedtuple (values, indices) where values is the minimum 
# # value of each row of the input tensor in the given dimension dim. 
# # And indices is the index location of each minimum value found (argmin).
# output_final, _ =torch.min(output, dim=2)
# np.sqrt(11**2+16+1+36+82)