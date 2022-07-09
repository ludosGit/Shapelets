import numpy as np
import numpy.random as random
import time
import sys
sys.path.append("/Documents/Shapelets_first_experiments")
from src import util
from tqdm import trange
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from src.util import euclidean_distance

####################
#  BRUTE FORCE SHAPELET EXTRACT FOR ANOMALY DETECTION
####################

class Candidateset():
    '''
    Class for storing a set of candidate subsequences together with position and score
    '''
    def __init__(self, sequences=np.array([]), positions=np.array([], dtype='int'), scores=np.array([])):
        self.sequences = np.array(sequences)
        self.positions = np.array(positions, dtype='int')
        self.scores = np.array(scores)

class Bruteforce_extractor():
    '''
    Class to extract the best shapelets from train_data
    '''
    def __init__(self, train_data, test_data, candidates=None, shapelets=None):
        '''
        train_data, test_data: bidimensional numpy arrays
        candidates, shapelets: objects from Candidateset
        '''
        self.train = train_data
        self.test = test_data
        self.candidates = candidates 
        self.shapelets = shapelets
    
    def extract_candidates(self, L_star=0.3):
        '''
        From train_data of shape (N, Q) extract all the candidates of length L_star * Q
        distance: distance measure for subsequnces of same length
        return: all the candidates as Candidateset object
        '''
        X = self.train
        N, Q = X.shape[0:2]
        L = round(L_star * Q)

        sequences = [] # transform later in numpy array
        positions = np.array([], dtype='int')
        scores = np.array([])

        for i in trange(N, desc='timeseries', position=0):
            for j in range(Q-L+1):
                S = X[i, j:j+L]
                sum = 0
                for index in range(N):
                    # sum all the sdists from every time series
                    sum += util.sdist(S, X[index,:])
                # append also the index of the position of the shapelet
                sequences.append(S)
                positions = np.append(positions, j)
                scores = np.append(scores, sum)
        sequences = np.array(sequences)
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
        self.shapelets = shapelets
        return shapelets

    def extract_shapelets(self, K_star=0.1, L_star=0.3, pos_boundary=0, reverse=False):
        '''
        Extract best shapelets from train_data
        :param X: ndarray of shape (N, Q, 1) with N time series all of the same length Q (can be modified for different lenghts)
        :param K_star: K = K_star * Q is the number of shapelets we want to discover
        :param L_star: L = L_star* Q is their length
        :distance: distance used for subsequnce similarity
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
        print('Time for shapelets extraction:')
        print("--- %s seconds ---" % (time.time() - start_time))
        return shapelets

    def plot_shapelets(self, path):
        '''
        Plot shapelets and save in path
        '''
        S = self.shapelets.sequences
    
        plt.figure()
        for i in range(len(S)):
            shap = S[i,:]
            plt.plot(shap, label=f'shapelet{i+1}')
            plt.legend()
            plt.title('The extracted shapelets', fontweight="bold")
        plt.savefig(path)
        return None

    def transform(self):
        '''
        Compute the shapelet tranform of both train and test data
        :return: X_train_transform, X_test_transform numy arrays of shape (N_train, K) and (N_test, K)
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
            shapelet = S[k, :]
            for i in range(N_train):
                T1 = X_train[i, :]
                d = util.sdist(shapelet, T1)
                X_train_transform[i, k] = d
            for j in range(N_test):
                T2 = X_test[j, :]
                d = util.sdist(shapelet, T2)
                X_test_transform[j, k] = d
        return X_train_transform, X_test_transform

# ##### TEST

# X = [[1,2,3,4,5], [6,7,8,9,10]]
# X = np.array(X)
# extractor = Bruteforce_extractor(data=X)
# K_star = 2/5
# L_star = 3/5
# shapelets = extractor.extract_shapelets(K_star, L_star)