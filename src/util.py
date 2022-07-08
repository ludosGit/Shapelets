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
    return scipy.linalg.norm(s1 - s2)

# ### TEST
# x = np.array([[4,5], [9,10]])
# y = np.array([[1,2], [1,2]])
# euclidean_distance(x, y)
# np.sqrt(146)


def length_normalized_distance(T1, T2):
    '''
    Returns ED between two time series of same length L, normalized by L
    :param T1: time series shape (L, C), with L length and C number of channels
    :param T2: time series shape (L, C)
    :return: euclidean length normalized distance
    NOTE: assume len(T1) == len(T2)
    '''
    # this works also with multivariate shapelets
    return np.sqrt(1/len(T1) * np.square(euclidean_distance(T1, T2)))

# ### TEST
# x = np.array([[1,2,3], [4,5,6]])
# len(x)
# x = np.transpose(x)
# y = np.array([[1,0,3], [4,0,6]])
# y = np.transpose(y)
# x = np.array([1, 0])
# y = np.array([0, 1])
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

################ DATA NORMALIZATION auxiliary class #############

class Normalizer():
    '''
    Class for normalize the time series 
    '''
    def __init__(self, scaler):
        '''
        scaler must be a scaler from sklearn.preprocessing
        '''
        self.scaler = scaler
        pass

    def fit_normalize(self, X):
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
    
    def normalize(self, X):
        'The scaler must be fitted before'
        shape = X.shape
        n_channels = shape[2]
        data_flat = X.reshape(-1, n_channels)
        data_transformed = self.scaler.transform(data_flat).reshape(shape)
        return data_transformed

# # multivariate test
# x = [np.array([[1,2,3,4,5], [6,7,8,9,10]]), np.array([[34,2,23,4,5], [6,72,81,9,10]])]
# x = np.array(x)
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



