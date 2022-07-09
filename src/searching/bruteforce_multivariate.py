import numpy as np
import numpy.random as random
import time
import sys
sys.path.append("/Documents/Shapelets_first_experiments")
from src import util
from src.util import euclidean_distance, sdist_mv
from tqdm import trange
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

####################
#  BRUTE FORCE SHAPELET EXTRACT FOR ANOMALY DETECTION MULTIVARIATE DATASET
####################

class Candidateset():
    '''
    Class for storing a set of candidate subsequences together with position and score
    '''
    def __init__(self, sequences=[], positions=[], scores=[]):
        self.sequences = np.array(sequences)
        self.positions = np.array(positions, dtype='int')
        self.scores = np.array(scores)

class Bruteforce_extractor_mv():
    '''
    Class to extract the best shapelets from multivariate train_data
    '''
    def __init__(self, train_data, test_data):
        '''
        train_data, test_data: bidimensional numpy arrays
        candidates, shapelets: objects from Candidateset
        '''
        self.train = train_data
        self.test = test_data
        self.candidates = None 
        self.shapelets = None
    
    def extract_candidates(self, L_star=0.3):
        '''
        From train_data of shape (N, Q, C), N number of time series, Q their length, extract all the candidates of length L_star * Q
        distance: distance measure for subsequnces of same length
        return: all the candidates as Candidateset object
        '''
        X = self.train
        N, Q = X.shape[0:2]
        L = round(L_star * Q)

        sequences = [] # transform later in numpy array
        positions = []
        scores = []

        for i in trange(N, desc='timeseries', position=0):
            for j in range(Q-L+1):
                S = X[i, j:j+L,:]
                sum = 0
                for index in range(N):
                    # sum all the SQUARED sdists from every time series
                    sum += util.sdist_mv(S, X[index,:,:])**2
                # append also the index of the position of the shapelet
                # don't use append numpy
                sequences.append(S)
                positions.append(j)
                # take the mean
                scores.append(sum / N)
        sequences = np.array(sequences)
        positions = np.array(positions, dtype='int')
        scores = np.array(scores)
        candidates = Candidateset(sequences, positions, scores)
        self.candidates = candidates
        return candidates

    def get_top_candidates(self, K, pos_boundary, reverse=False):
        '''
        Extract best K shapelets from self.candidates according to score in normal or reverse order
        with constraint distance(S_i, S-j) < threshold and |pos_i - pos_j| < pos_boundary
        return: shapelets as object from Candidateset
        '''
        candidates = self.candidates
        scores = candidates.scores
        assert scores is not None, 'Scores must be calculated'

        sequences = candidates.sequences # numpy array
        positions = candidates.positions

        indexes = scores.argsort()
        if reverse:
            indexes = indexes[::-1]
        
        sequences = sequences[indexes]
        positions = positions[indexes]
        scores = scores[indexes]

        # take best scoring shapelet

        seq_final = [sequences[0]]
        positions_final = [positions[0]]
        scores_final = [scores[0]]
        k = 0
        while len(seq_final) != K:
            S1 = seq_final[k]
            pos = positions_final[k]
            similarity_distances = []
            # iterate over all the candidates:
            for p in range(len(sequences)):
                S2 = sequences[p]
                d = euclidean_distance(S1,S2)
                # p is the index of the subsequence and d its distance to the last discovered shapelet
                similarity_distances.append(d)
            similarity_distances = np.array(similarity_distances)
            similarity_boundary = 0.1 * np.median(similarity_distances)
            # eliminate those candidates that don't satisfy the constraints

            indexes = np.logical_or((similarity_distances < similarity_boundary), (abs(positions - pos) < pos_boundary))
            sequences = np.delete(sequences, indexes, axis=0)
            positions = np.delete(positions, indexes, axis=0)
            scores = np.delete(scores, indexes, axis=0)

            seq_final.append(sequences[0])
            positions_final.append(positions[0])
            scores_final.append(scores[0])
            k += 1
        shapelets = Candidateset(seq_final, positions_final, scores_final)
        return shapelets

    def extract_shapelets(self, K_star=0.1, L_star=0.3, pos_boundary=0, reverse=False):
        '''
        Extract best shapelets from train_data
        :param X: ndarray of shape (N, Q, 1) with N time series all of the same length Q (can be modified for different lenghts)
        :param K_star: K = K_star * Q is the number of shapelets we want to discover
        :param L_star: L = L_star* Q is their length
        :reverse: whether to select the shapelets in reverse order, aka from the one that has highest sum of distances to the lowest
        :return: shapelets
        '''

        start_time = time.time()
        X = self.train
        N, Q = X.shape[0:2]
        L = round(L_star * Q)
        K = round(K_star * Q)
        print(f'Are going to be extracted {K:.3f} shapelets of length {L:.4f}')

        if reverse == True :
            print('Shapelets are going to be extracted in reverse order!')

        self.extract_candidates(L_star)
        shapelets = self.get_top_candidates(K, pos_boundary, reverse)
        self.shapelets = shapelets
        print('Time for shapelets extraction:')
        print("--- %s seconds ---" % (time.time() - start_time))
        return shapelets

    def transform(self):
        '''
        Compute the shapelet tranform of both train and test data
        :return: X_train_transform, X_test_transform 2D numy arrays of shape (N_train, K) and (N_test, K)
        '''
        assert self.shapelets is not None, 'Extract shapelets before'
        X_train = self.train
        X_test = self.test
        S = self.shapelets.sequences
        N_train = len(X_train)
        N_test = len(X_test)
        K = len(S)
        X_train_transform = np.zeros((N_train, K))
        X_test_transform = np.zeros((N_test, K))
        for k in range(K):
            shapelet = S[k,:,:]
            for i in range(N_train):
                T1 = X_train[i,:,:]
                d = util.sdist_mv(shapelet, T1)
                X_train_transform[i, k] = d
            for j in range(N_test):
                T2 = X_test[j,:,:]
                d = util.sdist_mv(shapelet, T2)
                X_test_transform[j, k] = d
        return X_train_transform, X_test_transform

# ##### TEST

# X = [np.array([[1,2,3,4,5], [6,7,8,9,10]]), np.array([[34,2,23,4,5], [6,72,81,9,10]])]
# X = np.array(X)
# X.shape
# X[1].shape
# X = np.moveaxis(X, -1, 1)


# extractor = Bruteforce_extractor_mv(train_data=X, test_data=None)
# K_star = 2/5
# L_star = 3/5
# shapelets = extractor.extract_shapelets(K_star, L_star)
# shapelets.positions
# X