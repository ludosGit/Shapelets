import numpy as np
import numpy.random as random
import time
import sys
sys.path.append("/Users/test1/Documents/TESI_Addfor_Industriale/Python_Projects_Shapelets/Shapelets_first_experiments-search_position")
from trials import util
from tqdm import trange
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from trials.util import euclidean_distance
import itertools

###################
#  Implement random local search algorithm
# idea is to substitute the methods which modify self with methods that create a new object
###################

class RLS_candidateset():
    def __init__(self, sequences=[], positions=np.array([], dtype='int'), lengths=np.array([], dtype='int'), scores=np.array([])):
        '''
        sequences is a list of numpy arrays of the sequences extracted
        '''
        assert (sequences==None or len(sequences) ==  len(positions) == len(lengths)), "Error: the information don't match"
        self.sequences = sequences
        self.positions = positions
        self.lengths = lengths
        self.scores = scores

    def __len__(self):
        return len(self.sequences)

    def is_empty(self):
        return self.sequences is None or len(self.sequences)==0

    def delete(self, indexes):
        '''
        Delete indexes from self
        indexes: list or numpy array of integers
        '''
        new_seq = [seq for i,seq in enumerate(self.sequences) if i not in indexes]
        new_positions = np.delete(self.positions, indexes)
        new_lengths = np.delete(self.lengths, indexes)
        new_set = RLS_candidateset(new_seq, new_positions, new_lengths)
        return new_set

    def get_neighborhood(self, j , L, epsilon):
        '''
        Take the subsequences that (satisfy position - j) < eps[0] e |length -L| < eps[1]: create a new RLS object 
        and ELIMINATE those sequences from self. Don't take into consideration the scores
        Parameters:
        j: position in the time series dataset
        L: length
        return: N neighborhood of the sequence as RLS_candidateset object
        '''
        assert (self.scores is None or len(self.scores)==0), "You can get a neighborhood only for a candidate set with empty scores"
        N = RLS_candidateset()
        # boolean array: True means satisky the condition
        indexes = np.logical_and(abs(self.positions -j) <= epsilon[0], abs(self.lengths - L) <= epsilon[1])
        # this command takes the index with true value from self.sequences
        N.sequences = list(itertools.compress(self.sequences, indexes))
        N.positions = self.positions[indexes]
        N.lengths = self.lengths[indexes]

        # now eliminate the sequences in N from self:
        indexes = np.ndarray.flatten(np.argwhere(indexes==True))
        new_set = self.delete(indexes)
        return N, new_set
    
    def set_scores(self, scores):
        self.scores = scores
        return None

    def merge(self, N):
        '''
        two candidate sets can be merged only if the score is computed for both
        '''
        new_sequences = extend(N.sequences)
        self.positions = np.append(self.positions, N.positions)
        self.lengths = np.append(self.lengths, N.lengths)
        self.scores = np.append(self.scores, N.scores)
        return self


# ##### TEST #######

# x = [[1,2,3], [4,5,6], [56]]
# y = [[7,8,9], [10, 11, 12, 13]]
# posx = [0, 2, 1]
# posy = [3, 0]   
# lenx = [3,3, 1]
# leny=  [3, 4]
# scorex = [1,2, 2]
# scorey = [2,3]
# X = RLS_candidateset(x, posx, lenx, scorex)
# Y = RLS_candidateset(y, posy, leny, scorey)

# X.merge(Y)
# X.sequences
# X.positions
# X.lengths
# X.is_empty()

# epsilon = (2,2)
# indexes = np.logical_and(abs(X.positions - 0) < epsilon[0], abs(X.lengths - 3) < epsilon[1])
# indexes = np.argwhere(indexes==True)
# indexes = np.ndarray.flatten(indexes)
# X.delete(indexes)

# X.scores = 6
# X.set_scores(None)

# N = X.get_neighborhood(0, 3, epsilon)
# X.sequences
# N.sequences
# N.lengths
# N.positions
# N.scores

class RLS_extractor:
    def __init__(self, data, candidates_notscored=None, candidates_scored=None):
        '''
        data: time series unlabeled dataset
        shapelets: best scoring candidates
        candidates: the ones I want to compute the scores
        total_candidates: all the sequences extracted from the data
        all are objects from RLS candidateset
        '''
        self.data = data
        self.candidates_notscored = candidates_notscored
        self.candidates_scored = candidates_scored
        self.shapelets = None

    def get_candidates(self, L):
        '''
        extract all the subsequences with length L
        return: 
        '''
        X = self.data
        N, Q = X.shape[0:2]

        seq = []
        positions = np.array([], dtype='int')
        lengths = np.array([], dtype='int')
        for i in range(N):
            for j in range(Q-L+1):
                S = X[i, j:j+L]
                # append also the index of the position of the shapelet
                # and index for the length
                seq.append(S)
                positions = np.append(positions, j)
                lengths = np.append(lengths, L)
            # add all the extracted info to total
        total_candidates = RLS_candidateset(seq, positions, lengths, scores=None)
        return total_candidates

    def get_random_candidates(self, r, L_star_min=0.3, L_star_max=None):
        '''
        Extract r random candidates of length from L_min to L_max included, searching from the subsequences in the dataset X
        Update all the candidates not scored in self.candidates_notscored
        :param X: ndarray of shape (N, Q, 1) with N time series all of the same length Q (can be modified for different lenghts)
        :param r: number of candidates to be selected randomly
        :param epsilon: must be a tuple of length 2
                        - epsilon[1] is the interval of neiborhood for the position
                        - epsilon[2] is the interval of neiborhood for the length
        :param K_star: K = K_star * Q is the number of shapelets we want to discover
        :param L_star_min: L_min = L_star_min* Q is their minimum length
        :param L_star_max: L_max = L_star_max* Q is their maximum length
        :distance: distance used for shapelets similarity
        :reverse: whether to select the shapelets in reverse order, aka from the one that has highest sum of distances to the lowest
        :return: candidates_notscored and random_candidates as objects from RLS_candidates class
        '''
        X = self.data
        N, Q = X.shape[0:2]
        L_min = round(L_star_min * Q)
        if L_star_max==None:
            L_max = L_min
        else:
            L_max = round(L_star_max * Q)

        seq_random = []
        positions_random = np.array([], dtype='int')
        lengths_random = np.array([], dtype='int')
        r_length = int(r / (L_max - L_min +1)) 

        total_seq = []
        total_positions = np.array([], dtype='int')
        total_lengths = np.array([], dtype='int')
        # r_length number of random candidates for each length
        # extract r candidates subsequences equally divided in number w.r.t lengths
        for L in range(L_min, L_max+1):
            # get all candidates of length L
            candidates = self.get_candidates(L)

            # take random candidates per length and append everything to the random vectors
            N_candidates = len(candidates.sequences)
            random_sample = random.choice(range(N_candidates), size=r_length, replace=False)
            seq_random.extend([candidates.sequences[i] for i in random_sample])
            positions_random = np.append(positions_random, candidates.positions[random_sample])
            lengths_random = np.append(lengths_random, candidates.lengths[random_sample])

            # delete from candidates the sequences extracted randomly whose score will be computed:
            candidates = candidates.delete(random_sample)

            # append everything to the total sequences
            total_seq.extend(candidates.sequences)
            total_positions = np.append(total_positions, candidates.positions)
            total_lengths= np.append(total_lengths, candidates.lengths)

        candidates_notscored = RLS_candidateset(total_seq, total_positions, total_lengths, scores=None)
        random_candidates = RLS_candidateset(seq_random, positions_random, lengths_random, scores=None)

        # update candidates info
        self.candidates_notscored = candidates_notscored
        self.candidates_scored = random_candidates
        print('the length of the total random sample should be equal to r', len(random_candidates))
        return candidates_notscored, random_candidates
        
    def compute_scores(self, candidates, distance=euclidean_distance):
        '''
        :candidates: object from RLS_candidaset class
        return: the same object with attribute score calculated as sum of sdists
        '''
        scores = np.array([])
        X = self.data
        N = len(X)
        r = len(candidates)
        for i in range(r):
            S = candidates.sequences[i]
            sum = 0
            for index in range(N):
                # sum all the sdists from every time series
                sum += util.sdist(S, X[index,:], distance)
                # append also the index of the position of the shapelet
            scores = np.append(scores, sum)
        candidates.scores = scores     
        return candidates

    def get_top_candidates(self, candidates_scored, m, pos_boundary=0, reverse=False, distance=euclidean_distance):
        '''
        Extract best m best candidates in a RLS_candidateset object with computed scores
        :param X: ndarray of shape (N, Q) with N time series all of the same length Q (can be modified for different lenghts)
        :return: 
        !!!!NOTE: the euclidean distance contraint is valid only if they have the same length!!!!!
        in case use different lengths it must be changed
        '''
        scores = candidates_scored.scores
        assert scores is not None, 'Must calculate scores before'
        if len(candidates_scored) < m:
            print('Error: too few candidates')
            return None
        
        indexes = scores.argsort()
        if reverse:
            indexes = indexes[::-1]
        subsequences = [candidates_scored.sequences[i] for i in indexes]
        # print('ordered subse', subsequences)
        # print('the scores', scores)
        positions = candidates_scored.positions[indexes]
        lengths = candidates_scored.lengths[indexes]
        scores = scores[indexes]

        # take the first one
        best_subsequences = [subsequences[0]]
        # print('best', best_subsequences)
        # print('len', len(best_subsequences))
        best_positions = np.array([positions[0]])
        best_lengths = np.array([lengths[0]])
        best_scores = np.array([scores[0]])
        k = 0
        while len(best_subsequences) != m or len(subsequences)==0:
            S1 = best_subsequences[k]
            pos = best_positions[k]
            similarity_distances = []
            for p in range(len(subsequences)):
                S2 = subsequences[p]
                d = distance(S1,S2)
                # p is the index of the subsequence and d its distance to the last discovered shapelet
                similarity_distances.append(d)
            # this works only if candidates have the same length!!
            similarity_distances = np.array(similarity_distances)
            # print('dist', similarity_distances)
            similarity_boundary = 0.1 * np.median(similarity_distances)
            # print('boundary', similarity_boundary)
            # eliminate those shapelets too similar from the one considered
            indexes = np.logical_or((similarity_distances < similarity_boundary), (abs(pos - positions) < pos_boundary))
            # print('ind', indexes)

            # delete the indexes in the subsequences taking into account it is a list!!
            subsequences_removed = []
            for i in range(len(subsequences)):
                if not indexes[i]:
                    subsequences_removed.append(subsequences[i])
            subsequences = subsequences_removed

            # delete the others using numpy
            positions = np.delete(positions, indexes, axis=0)
            scores = np.delete(scores, indexes, axis=0)
            lengths = np.delete(lengths, indexes, axis=0)

            best_subsequences.append(subsequences[0])
            best_positions = np.append(best_positions, positions[0])
            best_scores = np.append(best_scores, scores[0])
            best_lengths = np.append(best_lengths, lengths[0])
            k += 1
        
        best_candidates = RLS_candidateset(best_subsequences, best_positions, best_lengths, best_scores)
        return best_candidates


    def LocalSearch(self, best_candidates, epsilon = (1,1), reverse=False):
        '''
        Perform a local search in candidate space:
        best_candidates: object from RLS_candidateset class
        for each candidate search in self.candidates_notscored the neighbors
        '''
        candidates_notscored = self.candidates_notscored

        if reverse:
            # worst_score = min(best_candidates.scores)
            for i in range(len(best_candidates)):
                print(f'Searching in sequence number {i} neighbors')
                j = best_candidates.positions[i]
                L = best_candidates.lengths[i]
                score = best_candidates.scores[i]
                
                while True:
                    # the method get_neighborhood eliminates N from candidates_notscored
                    N = candidates_notscored.get_neighborhood(j, L, epsilon)

                    # compute scores and merge with scored candidates
                    N = self.compute_scores(N)
                    self.candidates_scored = self.candidates_scored.merge(N)
                    # max is the best score in the neighborhood if reverse is true
                    # with this condition you can accidentally discover better candidates
                    if (N.is_empty() or max(N.scores) <= score):
                        # print(f'bad search, the sequences {N.sequences} have scores {N.scores}')
                        break
                    best_candidates = best_candidates.merge(N)

                    # print('neighbors found', best_candidates.sequences)
                    index = np.argwhere(N.scores ==max(N.scores))[0]

                    #update j and L to search 
                    j = N.positions[index]
                    L = N.lengths[index]
            return best_candidates 
                   
        else:
            worst_score = max(best_candidates.scores)
            for i in range(len(best_candidates)):
                print(f'Searching in sequence number {i} neighbors')
                while True:
                    j = best_candidates.positions[i]
                    L = best_candidates.lengths[i]
                    N = candidates_notscored.get_neighborhood(j, L, epsilon)


                    N = self.compute_scores(N)
                    self.candidates_scored = self.candidates_scored.merge(N)
                    # min is the best score in the neighborhood if reverse is false
                    if N.is_empty() or min(N.scores) >= worst_score:
                        # print(f'bad search the sequences {N.sequences} have scores {N.scores}')
                        break
                    best_candidates = best_candidates.merge(N)
                    # print('neighbors found', best_candidates.sequences)

                    index = np.argwhere(N.scores ==min(N.scores))[0]
                    #update j and L to search 
                    j = N.positions[index]
                    L = N.lengths[index]
            return best_candidates  
        

    def extract(self, r, m, pos_boundary=0, epsilon=(1,1), reverse=False, K_star = 0.02, L_star_min=0.2, L_star_max=None):
        N, Q = self.data.shape[0:2]
        K = round(K_star*Q)

        _, random_candidates = self.get_random_candidates(r, L_star_min, L_star_max)
        print('finished to get random candidates')
        self.compute_scores(random_candidates, distance=euclidean_distance)
        best_candidates = self.get_top_candidates(random_candidates, m, pos_boundary, reverse, distance=euclidean_distance)
        print(f'finished to get top {m} candidates with scores {best_candidates.scores}')
        best_candidates = self.LocalSearch(best_candidates, epsilon, reverse)
        print('finished local search')
        shapelets = self.get_top_candidates(best_candidates, K, pos_boundary, reverse, distance=euclidean_distance)
        print(f'finished to get top {K} candidates')
        self.shapelets = shapelets
        return shapelets

# ### TEST #######

# data = np.array([[122,278,30,3], [465,45,62,4], [270,80,9,5]])
# extractor = RLS_extractor(data=data)
# extractor.data
# x = extractor.get_candidates(2)
# x.sequences
# x.positions
# len(x)

# not_scored, rand = extractor.get_random_candidates(r=4, L_star_min=2/4)
# not_scored.sequences # list of numpy arrays!!
# rand.sequences
# extractor.compute_scores(rand)
# rand.scores
# best = extractor.get_top_candidates(rand, 2)
# best.positions
# best.scores
# best = extractor.LocalSearch(best)
# best.sequences
# best.scores
# shapelets = extractor.get_top_candidates(best, 2)
# shapelets.sequences
# shapelets.scores

# extractor.candidates_notscored.sequences
# extractor.candidates_scored.sequences