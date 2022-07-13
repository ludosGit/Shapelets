import numpy as np
import scipy
import tslearn
from tslearn import preprocessing
from tslearn.datasets import CachedDatasets

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


################### DISTANCE FUNCTIONS IMPLEMENTATION ############

def euclidean_distance(s1, s2):
    '''
    Compute the Frobenious norm of the matrix s1-s2 (with square root)
    :s1, s2: numpy arrays of same shape
    '''
    return np.linalg.norm(s1 - s2)

# ### TEST
# x = np.array([[4,5], [9,10]])
# y = np.array([[1,2], [1,2]])
# euclidean_distance(x, y)
# np.sqrt(146)

def normalize_1d(x):
    return (x - np.mean(x)) / np.std(x)

def normalize_2d(x):
    '''
    normalize x channel wise
    @param x: time series shape (n_observations, n_channels)
    '''
    return (x - np.mean(x, axis=0, keepdims=True)) # / np.std(x, axis=0, keepdims=True)

# ### TEST
# x = np.array([[1,2,3], [4,5,6]]).transpose(1,0)
# normalize_2d(x)
# x = np.array([1,2,3])
# (x - 2) / np.std(x)

def length_normalized_distance(t1, t2):
    '''
    Returns ED between two time series of same length L, normalized by L
    NOTE: both the series are shifted by their channel-wise mean
    :param t1: time series shape (L, C), with L length and C number of channels
    :param t2: time series shape (L, C)
    :return: euclidean length normalized distance
    NOTE: assume len(t1) == len(t2)
    '''
    # this works also with multivariate shapelets
    # return np.sqrt(1/len(t1) * np.square(euclidean_distance((t1 - np.mean(t1, axis=0, keepdims=True)),(t2 - np.mean(t2, axis=0, keepdims=True)) )))
    return 1/len(t1) * euclidean_distance(normalize_2d(t1), normalize_2d(t2))

# ### TEST
# x = np.array([[1,2,3], [4,5,6]])
# len(x)
# x = np.transpose(x)
# x - np.mean(x, axis=0, keepdims=True)

# y = np.array([[1,0,3], [4,0,6]])
# y = np.transpose(y)
# x = np.array([1, 1]).reshape(-1,1)
# y = np.array([20, 20]).reshape(-1,1)

# length_normalized_distance(x, y)


def sdist(S, T):
    ''' 
    Compute l2-discrepancy (subsequence distance) between shapelet S and time series T 
    NOTE: implementation for univariate time series 
    :S: single shapelet, numpy array shape (L,)
    :T: a time series, numpy array shape (Q,)
    :return: real number of min subsequence distance
    '''
    Q = len(T)
    L = len(S)

    # if the shapelet and time series are exchanged, swap them:
    if L > Q:
        Z = T
        T = S
        S = Z
        Q = len(T)
        L = len(S)
    
    num_seg = Q - L + 1
    D2 = np.zeros((num_seg))
    for q in range(num_seg):
        seg = T[q:q + L] # q th segment of long series
        D2[q] = length_normalized_distance(seg, S)
    return np.min(D2)

# ## TEST
# S = np.array([1,2,3])
# T = np.array([2,3,3,4,5])
# sdist(S,T)
# sdist(T,S)
# T = np.array([2,3,3])
# sdist(S,T)
# length_normalized_distance(S,T)
# np.sqrt(sdist(S,T)**2*3)
# euclidean_distance(S,T)

def sdist_mv(S, T):
    '''
    Compute l2-discrepancy (subsequence distance) between shapelet S and time series T 
    :S: single shapelet of shape (L,C), L length and C number of channels
    :T: a time series of shape (Q,C), Q length and C number of channels
    :return: real number of min subsequence distance
    '''
    Q = len(T)
    L = len(S)
    # if the shapelet and time series are exchanged, swap them:
    if L > Q:
        Z = T
        T = S
        S = Z
        Q = len(T)
        L = len(S)

    num_seg = Q - L + 1
    D2 = np.zeros((num_seg))
    for q in range(num_seg):
        seg = T[q:q + L,:] # q th segment of long series
        D2[q] = length_normalized_distance(seg, S)
    return np.min(D2)

# ## TEST
# S = np.array([1,2,3])
# S.size
# T = np.array([2,3,3,4,5])
# S = S.reshape(3,1)
# T = T.reshape(5,1)
# sdist_mv(S,T)
# sdist_mv(T,S)
# S.shape
# T = np.array([2,3,3])
# sdist(S,T)
# length_normalized_distance(S,T)
# np.sqrt(sdist(S,T)**2*3)
# euclidean_distance(S,T)

############### CROSS CORRELATION 

def xcorr(x, y, scale='biased'):
    # Pad shorter array if signals are different lengths
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))

    corr = np.correlate(x, y, mode='full')  # scale = 'none'
    lags = np.arange(-(x.size - 1), x.size)

    if scale == 'biased':
        corr = corr / x.size
    elif scale == 'unbiased':
        corr /= (x.size - abs(lags))
    elif scale == 'coeff':
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    return corr

# # TEST
# x = np.array([1,1,1,1,5,5,5,1,1])
# mu_x = np.mean(x)
# sd_x = np.std(x)
# x = (x - mu_x) / (sd_x)

# y = np.array([1,5,5,5,1,1,1,1,1])
# mu_y = np.mean(y)
# sd_y = np.std(y)
# y = (y - mu_y) /sd_y
# z = xcorr(x,y)
# z = np.correlate(x,y, mode='full')
# max(z)


def max_corr(x, y, scale='biased'):
    '''
    @param x,y: numpy arrays shape (n_observations, n_channels)
    return: average of the max cross correlations of x and y channelwise
    '''
    n_channels = x.shape[1]
    tmp = np.zeros(n_channels)
    for c in range(n_channels):
        tmp[c] = max(xcorr(normalize_1d(x[:,c]), normalize_1d(y[:,c]), scale=scale))
    return np.mean(tmp)

# # TEST
# x = np.array([[1,2,3,1,1], [1,1,1,5,5]]).transpose(1,0)
# y = np.array([[1,1,1,2,3], [1,1,1,5,5]]).transpose(1,0)
# z = max_corr(x,y)
# z
# tmp = np.zeros(2)
# for c in range(2):
#     print(x[:,c].size)
#     tmp[c] = max(xcorr(x[:,c], y[:,c], scale='biased'))
# z = xcorr(normalize_1d(x[:,0]), normalize_1d(y[:,0]), scale='biased')
# z = np.correlate(normalize_1d(x[:,1]), normalize_1d(y[:,1]), mode='full')
# z = z / 5
# max(z)






################ DATA NORMALIZATION auxiliary class #############
# in order to normalize a dataset globally

class Scaler():
    '''
    Class for normalize the time series 
    '''
    def __init__(self, scaler):
        '''
        scaler must be a scaler from sklearn.preprocessing
        '''
        self.scaler = scaler
        pass

    def fit_transform(self, X):
        '''
        @param X: time series np array shape (n_samples, len_samples, n_channels)
        return: data normalized per channel
        '''
        shape = X.shape
        n_channels = shape[2]
        data_flat = X.reshape(-1, n_channels)
        if self.scaler is None:
            self.scaler = StandardScaler()
            data_transformed = self.scaler.fit_transform(data_flat).reshape(shape)
        else:
            data_transformed = self.scaler.fit_transform(data_flat).reshape(shape)
        return data_transformed
    
    def transform(self, X):
        'The scaler must be fitted before'
        shape = X.shape
        n_channels = shape[2]
        data_flat = X.reshape(-1, n_channels)
        data_transformed = self.scaler.transform(data_flat).reshape(shape)
        return data_transformed

# # multivariate test
# x = [np.array([[1,2,3,4,5], [6,7,8,9,10]]), np.array([[34,2,23,4,5], [6,72,81,9,10]])]
# x = np.array(x)
# x[0].size
# len(x[0])
# x = np.moveaxis(x, -1, 1)
# x = x.reshape((-1,2))
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)


################### PERMISSIBLE ERRORS IMPLEMENTATION ####################

def pairs_shapelet_index(t, S):
    '''
    Compute the pairs (k, j), which indicate that the shapelet k best approximates t at position j
    :param S:
    :param t:
    :return: triplets (d, k, j)
    '''
    Q = len(t)
    K, L = S.shape[0:2]
    pairs = []
    j = 0
    ### CASE j = 0:
    t_j = t[j: j+L]
    D = []
    for k in range(K):
        d = length_normalized_distance(t_j, S[k, :])
        D.append((d, k, j))
    D = sorted(D, key= lambda x: x[0], reverse=False)
    pairs.append(D[0])

    ### CASE 0 < j < Q-L
    while j != Q-L:
        D = []
        for start in range(1, L+1):
            end = min(j + start + L, Q)
            t_j = t[(j + start) : end]
            # t_j has length L in the normal case
            for k in range(K):
                d = length_normalized_distance(t_j, S[k,:])
                D.append((d, k, j+start))
        D = sorted(D, key= lambda x: x[0], reverse=False)
        d, k, j = D[0]
        # means the shapelet k best approximates locally (starting at j) the time series t with distance d
        pairs.append((d, k, j))

    # ### CASE j = Q-L (pretty sure does it automatically)
    # t_j = t[j: j+L]
    # D = []
    # for k in range(K):
    #     d = length_normalized_distance(t_j, S[k, :])
    #     D.append((d, k, j))
    # D = sorted(D, key= lambda x: x[0], reverse=False)
    # pairs.append(D[0])
    return pairs

def permissible_error_transform(t, S):
    '''
    :param t: A single time series
    :param S: A SET of shapelets
    :param pairs: output of 'pairs_shapelet_index': is actually a list of triplets (distance, shapelet, position)
    which is peculiar of each time series and sets of shapelets
    :return: x vector of length K, is the permissible error of t w.r.t. all the shapelets
    '''
    K, L = S.shape
    pairs = pairs_shapelet_index(t, S)
    x = []
    for k in range(K):
        s = S[k, :]
        pairs_k = list(filter(lambda x: x[1]==k, pairs))
        if len(pairs_k) != 0:
            pairs_k = sorted(pairs_k, key= lambda x: x[0], reverse=True)
            # append max value
            x.append(pairs_k[0][0])
        else:
            x.append(sdist(s, t))
    return x

# # ### test the permissible error functions
# t = np.array([i for i in range(10)])
# print(t)
# S = np.array([[0,1,2,3], [4, 5, 6, 7]])
# pairs = pairs_shapelet_index(t, S)
# print(pairs)
# x = permissible_error_transform(t, S, pairs)
# print(x)

########### GENDIS ADAPTATION ##############


def pairs_shapelet_index_gendis(t, S):
    '''
    Adapted to the shapelets type in GENDIS algorithm
    Compute the pairs (k, j), which indicate that the shapelet k best approximates t at position j
    :param S: set of shapelets
    :param t:
    :return: triplets (d, k, j)
    '''
    Q = len(t)
    K = len(S)
    L = len(S[0])
    # assume are all of same length
    pairs = []
    j = 0
    ### CASE j = 0:
    t_j = t[j: j+L]
    D = []
    for k in range(K):
        d = length_normalized_distance(t_j, S[k])
        D.append((d, k, j))
    D = sorted(D, key= lambda x: x[0], reverse=False)
    pairs.append(D[0])

    ### CASE 0 < j < Q-L
    while j != Q-L:
        D = []
        for start in range(1, L+1):
            end = min(j + start + L, Q)
            t_j = t[(j + start) : end]
            # t_j has length L in the normal case
            for k in range(K):
                d = length_normalized_distance(t_j, S[k])
                D.append((d, k, j+start))
        D = sorted(D, key= lambda x: x[0], reverse=False)
        d, k, j = D[0]
        # means the shapelet k best approximates locally (starting at j) the time series t with distance d
        pairs.append((d, k, j))
    return pairs






#################### SHAPELET CLUSTERING FUNCTIONS ##########################

def compute_gap(timeseries, D_A, D_B):
    distance_A = D_A['d']
    distance_B = D_B['d']
    pos_A = D_A['pos']
    pos_B = D_B['pos']
    mu_A = np.mean(distance_A)
    std_A = np.std(distance_A)
    mu_B = np.mean(distance_B)
    std_B = np.std(distance_B)
    gap = mu_B-std_B-mu_A-std_A
    threshold = mu_A + std_A
    return gap, threshold

def compute_best_gap(timeseries, S, k):
    orderline = []
    for i in range(len(timeseries)):
        T = timeseries[i]
        d = sdist(S,T)
        orderline.append((d,i)) # save the index
    orderline = sorted(orderline, key=lambda x: x[0], reverse=False)
    gap_list = []
    best_gap = 0
    best_threshold = 0
    for i in range(len(orderline)-1):
        # N-1 candidates thresholds
        candidate_dt = (orderline[i+1][0] - orderline[i][0])/2
        # D_A: indexes of the timeseries that match the shapelet wrt threshold candidate_dt
        D_A = {'pos': [], 'd': []}
        D_B = {'pos': [], 'd': []}
        for d, pos in orderline:
            if d < candidate_dt:
                D_A['d'].append(d)
                D_A['pos'].append(pos)
            else:
                D_B['d'].append(d)
                D_B['pos'].append(pos)
        if 1/k < len(D_A['d'])/len(D_B['d']) < (1-1/k):
            continue
        gap, threshold = compute_gap(timeseries, D_A, D_B)
        if gap > best_gap:
            best_gap= gap
            best_threshold = threshold
    return best_gap, best_threshold, orderline



