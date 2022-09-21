from collections import OrderedDict
import warnings
import sys
from zlib import Z_RLE
sys.path.append('../../')

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.learning.distancelayer import DistanceLayer
from src.learning.distanceloss import L2DistanceLoss, SVDD_L2DistanceLoss
from src.learning.similarityloss import CorrelationSimilairty
from src.SVDD import SVDD

################### CLASS for a shapelet - learning extractor 

class LearningShapelets():
    """
    Wraps Learning Shapelets in a sklearn kind of fashion.
    Parameters
    ----------
    len_shapelets:
    num_shapelets:
    in_channels : int
        the number of input channels of the dataset
    dist_measure: `euclidean`, `cross-correlation`, or `cosine`
        the distance measure to use to compute the distances between the shapelets.
      and the time series.
    verbose : bool
        monitors training loss if set to true.
    to_cuda : bool
        if true loads everything to the GPU
    """

    def __init__(self, len_shapelets, num_shapelets, in_channels=1, \
        radius=0, C=1, verbose=0, to_cuda=True, l1=0, loss_sim=CorrelationSimilairty()):
        '''
        @param radius: initial radius of SVDD boundary
        '''

        self.model = DistanceLayer(len_shapelets, num_shapelets, in_channels, to_cuda)
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.len_shapelets = len_shapelets
        self.num_shapelets = num_shapelets
        self.loss_func = SVDD_L2DistanceLoss(radius=radius)
        self.verbose = verbose
        self.optimizer = None
        self.scheduler = None
        self.C = C
        self.l1 = l1
        if self.l1 > 0.0:
            self.loss_sim = loss_sim

    def set_optimizer(self, optimizer):
        """
        Set an optimizer for training.
        @param optimizer: a PyTorch optimizer: https://pytorch.org/docs/stable/optim.html
        @type optimizer: torch.optim
        @return:
        @rtype: None
        """
        self.optimizer = optimizer
        return None

    def set_scheduler(self, scheduler):
        """
        Set an optimizer for training.
        @param optimizer: a PyTorch optimizer: https://pytorch.org/docs/stable/optim.html
        @type optimizer: torch.optim
        @return:
        @rtype: None
        """
        self.scheduler = scheduler
        return None

    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. 
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, len_shapelets)
        @return:
        @rtype: None
        """
        self.model.set_shapelet_weights(weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")

    def update(self, x):
        """
        Performs one gradient update step for the batch of time series and corresponding labels y.
        @param x: the batch of time series
        @type x: array-like(float) of shape (n_batch, in_channels, len_ts)
        @param y: the labels of x
        @type y: array-like(long) of shape (n_batch)
        @return: the loss for the batch
        @rtype: float
        """
        # forward pass
        x_transformed = self.model(x)
        loss_dist = self.loss_func(x_transformed)
        loss_dist.backward(retain_graph=True)
        s = self.model.get_shapelets()
        if self.l1 > 0.0:
            # get shapelet similarity loss and compute gradients
            loss_sim = self.loss_sim(s) * self.l1
            loss_sim.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss_dist.item(), loss_sim.item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss_dist.item()


    def compute_radius(self, X, tol=1e-6):
        X = self.transform(X) # numpy array shape (n_samples, n_shapelets)
        svdd = SVDD.SVDD(C=self.C, zero_center=True, verbose=False, tol=tol)
        svdd.fit(X)
        self.loss_func.update_r(svdd.radius)
        return None

    def fit(self, X, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        Train the model.
        @param X: the time series data set
        @type X: array-like(float) of shape (n_samples, in_channels, len_ts)
        @param Y: the labels of x

        # for now labels ignored, idea to include pseudo class labels
        @type Y: array-like(long) of shape (n_batch)
        @param epochs: the number of epochs to train

        @type epochs: int
        @param batch_size: the batch to train with
        @type batch_size: int
        @param shuffle: Shuffle the data at every epoch
        @type shuffle: bool
        @param drop_last: Drop the last batch if X is not divisible by the batch size
        @type drop_last: bool
        @return: a list of the training losses
        @rtype: list(float)
        """
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float).contiguous()
        # if not isinstance(Y, torch.Tensor):
        #     Y = tensor(Y, dtype=torch.long).contiguous()
        if self.to_cuda:
            X = X.cuda()

        train_ds = TensorDataset(X)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        
        # set model in train mode
        self.model.train()

        # create a list of the losses for each batch
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_dist = 0
        current_loss_sim = 0
        # print('shape of X before starting', X.shape)
        if self.l1 > 0:
            for _ in progress_bar: # for each epoch
                for j, x in enumerate(train_dl):
                    x = x[0] # items in dataloader are lists because a label is supposed
                    current_loss_dist, current_loss_sim = self.update(x)
                    losses_dist.append(current_loss_dist)
                    losses_sim.append(current_loss_sim)
                progress_bar.set_description(f"Loss dist: {current_loss_dist}")
                
            if self.scheduler is not None:
                self.scheduler.step()
            return losses_dist, losses_sim 
        # if no similarity loss:
        for _ in progress_bar: # for each epoch
            for j, x in enumerate(train_dl):
                x = x[0] # items in dataloader are lists because a label is supposed
                current_loss_dist = self.update(x)
                losses_dist.append(current_loss_dist)
            progress_bar.set_description(f"Loss dist: {current_loss_dist}")

        if self.scheduler is not None:
            self.scheduler.step()
        return losses_dist
        

    def transform(self, X):
        """
        Performs the shapelet transform with the input time series data x
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the shapelet transform of x
        @rtype: numpy array (float) of shape (num_samples, num_shapelets)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if self.to_cuda:
            X = X.cuda()

        with torch.no_grad():
            shapelet_transform = self.model.forward(X)
             # transform was implemented in learning shapelet model
            # simply ignores last distance layer
            # in our version we have only distance layer
        if self.num_shapelets == 1 and self.model.in_channels==1:
            return shapelet_transform.squeeze().cpu().detach().numpy().reshape((-1,1))
        return shapelet_transform.squeeze().cpu().detach().numpy()
        # sqeeze(): Returns a tensor with all the dimensions of input of size 1 removed
        # detach(): Returns a new Tensor, detached from the current graph. The result will never require gradient.
        # numpy(): Returns self tensor as a NumPy ndarray. 

    def fit_transform(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        fit() followed by transform().
        @param X: the time series data set
        @type X: array-like(float) of shape (n_samples, in_channels, len_ts)
        @param Y: the labels of x
        @type Y: array-like(long) of shape (n_batch)
        @param epochs: the number of epochs to train
        @type epochs: int
        @param batch_size: the batch to train with
        @type batch_size: int
        @param shuffle: Shuffle the data at every epoch
        @type shuffle: bool
        @param drop_last: Drop the last batch if X is not divisible by the batch size
        @type drop_last: bool
        @return: the shapelet transform of x
        @rtype: tensor(float) of shape (num_samples, num_shapelets)
        """
        self.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return self.transform(X)

    def predict(self, X, batch_size=256):
        """
        Use the model for inference.
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @param batch_size: the batch to predict with
        @type batch_size: int
        @return: the logits for the class predictions of the model
        @rtype: array(float) of shape (num_samples, num_classes)
        """
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # set model in eval mode
        self.model.eval()

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        with torch.no_grad():
            for x in dl:
                y_hat = self.model(x[0])
                y_hat = y_hat.cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: an array of all shapelets
        @rtype: numpy.array(float) with shape (num_total_shapelets, in_channels, shapelets_size_max)
        """
        return self.model.get_shapelets().clone().cpu().detach().numpy()