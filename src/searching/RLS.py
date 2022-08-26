import numpy as np
import numpy.random as random
import time
import sys
sys.path.append("/Users/test1/Documents/TESI_Addfor_Industriale/Python_Projects_Shapelets/Shapelets_first_experiments-search_position")
from src import util
from tqdm import trange
import matplotlib.pyplot as plt
from src.util import euclidean_distance, sdist_mv, euclidean_distance_shifted, max_corr, mean_shift
import itertools

###################
#  Implement random local search algorithm
###################

class RLS_candidateset():
    ''' Class for storing information about a set of subsequences in a time series dataset:
    including values, starting positions, lengths and scores'''

    def __init__(self, sequences=[], positions=[], lengths=[], scores=[]):
        '''
        sequences: list of numpy arrays of the sequences extracted
        positions: numpy array of their starting positions
        lenghts: numpy array of their lengths
        scores: numpy array of their scores (possibly empty)
        NOTE: sequences, positons and lengths must have the same lengths
        '''
        self.sequences = sequences
        self.positions = np.array(positions, dtype='int')
        self.lengths = np.array(lengths, dtype='int')
        self.scores = np.array(scores)

    def __len__(self):
        return len(self.sequences)

    def is_empty(self):
        return self.sequences is None or len(self.sequences)==0

    def delete(self, indexes):
        '''
        Delete indexes from self (modifying self NOT creating a new one)
        NOTE: self.scores must be empty
        Indexes represent the indexes of the subsequences to delete in self
        :indexes: list or numpy array of integers (not booleans) 
        '''
        # assume always sequences is a list
        new_seq = [seq for i,seq in enumerate(self.sequences) if i not in indexes]
        self.sequences = new_seq
        
        # while positions and lengths are numpy arrays
        self.positions = np.delete(self.positions, indexes)
        self.lengths = np.delete(self.lengths, indexes)
        return self

    def get_neighborhood(self, j , L, epsilon, step=1, beta=0):
        '''
        Take a percentage (1-beta)*100 of the subsequences that satisfy:
        |position - j| <= eps[0]
        |length - L| <= eps[1] * step
        and create a new RLS_candidateset object with those data
        ELIMINATE those sequences from self
        NOTE: self.scores must be empty
        Parameters:
        :j: position in the time series dataset, integer
        :L: length, integer
        :epsilon: neighborhood range values
        :step: step of lengths starting from L_min 
        :beta: percentage of neighbors to discard 
        return: N neighborhood of the sequence as RLS_candidateset object
        '''
        assert (self.scores is None or len(self.scores)==0), "You can get a neighborhood only for a candidate set with empty scores"

        # create empty set to store the neighbors
        N = RLS_candidateset()

        # boolean array: True means satisfy the condition
        indexes_bool = np.logical_and(abs(self.positions -j) <= epsilon[0], abs(self.lengths - L) <= epsilon[1] * step)
        indexes = np.ndarray.flatten(np.argwhere(indexes_bool==True))
        A = len(indexes)
        sample_size = round(A * beta)
        to_eliminate = indexes[random.choice(A, size=sample_size, replace=False)]

        # set False the indices I don't want to calculcate the score
        indexes_bool[to_eliminate] = False

        # UPDATE the indexes: the new indexes are less than before because I set randomly some equal to False
        # in order to save computations
        indexes = np.ndarray.flatten(np.argwhere(indexes_bool==True))

        # Take the sequences with True indexes and store them in N

        # this command takes the index with true value from self.sequences
        N.sequences = list(itertools.compress(self.sequences, indexes_bool))
        N.positions = self.positions[indexes_bool]
        N.lengths = self.lengths[indexes_bool]

        # now eliminate the sequences in N from self:
        self.delete(indexes)
        return N

# ########## TEST GET_NEIGHBORHOOD
# x = np.array([True, False, True, False, False])
# indexes = np.ndarray.flatten(np.argwhere(x==True))
# A = len(indexes)
# sample_size = round(A*0.5)
# to_eliminate = indexes[random.choice(A, size=sample_size, replace=False)]

# # set False the indices I don't want to calculcate the score
# x[to_eliminate] = False
# x
# # update the indexes
# indexes = np.ndarray.flatten(np.argwhere(x==True))
    
    def set_scores(self, scores):
        self.scores = scores
        return None

    def merge(self, N):
        '''
        Add to self all the attributes of N, modify self
        N: RLS_candidateset object
        NOTE: two candidate sets can be merged only if the score is computed for both
        '''
        self.sequences.extend(N.sequences)
        self.positions = np.append(self.positions, N.positions)
        self.lengths = np.append(self.lengths, N.lengths)
        self.scores = np.append(self.scores, N.scores)
        return self

# tested ok
    def random_sample(self, sample_size=3000):
        '''
        Select random sequences according to the proportion prop
        :return: new RLS_candidateset
        '''
        N = len(self)
        random_sample = random.choice(range(N), size=sample_size, replace=False)
        random_sequences = [self.sequences[i] for i in random_sample]
        positions = self.positions[random_sample]
        lengths = self.lengths[random_sample]
        R = RLS_candidateset(random_sequences, positions, lengths)
        return R



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

# epsilon = (1,1)
# indexes = np.logical_and(abs(X.positions - 0) <= epsilon[0], abs(X.lengths - 3) <= epsilon[1])
# indexes = np.argwhere(indexes==True)
# indexes = np.ndarray.flatten(indexes)
# X.delete(indexes)

# X.scores = 6
# X.set_scores(None)

# N = X.get_neighborhood(0, 3, epsilon, alpha=0.5)
# X.sequences
# N.sequences
# N.lengths
# N.positions
# N.scores


class RLS_extractor():
    '''
    Class for extracting shapelets from train_set with Random Local Search.
    As more subsequences are calculated the scores, the attribute candidates_scored
    becomes larger, while candidates_notscored smaller.
    Their sum is the total number of subsequences in the train_data in the predefined range of lengths
    '''

    def __init__(self, train_data, test_data):
        '''
        data: time series unlabeled dataset, numpy array with shape (N,Q,C)
        candidates_notscored: candidates for whom a score is not yet computed (RLS candidateset)
        candidates_scored: the ones with computed scores (RLS candidateset)
        '''
        self.data = train_data
        self.test_data = test_data
        self.candidates_notscored = None
        self.candidates_scored = None
        self.total_candidates = None
        self.shapelets = None
        self.L_min = None
        self.L_max = None

    def get_candidates(self, L):
        '''
        extract all the subsequences with length L and empty scores
        return: RLS_candidateset
        '''
        X = self.data
        N, Q = X.shape[0:2]

        seq = []
        positions = []
        lengths = []
        for i in range(N):
            for j in range(Q-L+1):
                S = X[i, j:j+L, :]
                # append also the index of the position of the shapelet
                # and index for the length
                seq.append(S)
                positions.append(j)
                lengths.append(L)
            # add all the extracted info to total
        positions = np.array(positions, dtype='int')
        lengths = np.array(lengths, dtype='int')
        total_candidates = RLS_candidateset(seq, positions, lengths, scores=[])
        return total_candidates

    def get_random_candidates(self, r, L_min, step=1, n_steps=0):
        '''
        Extract r random candidates of length from L_min to L_max included, 
        searching from the subsequences in self.data (train_data)
        Compute their scores and update self.candidates_scored
        NOTE: Update also all the candidates not scored in self.candidates_notscored

        :param X: ndarray of shape (N, Q, C) with N time series all of the same length Q (can be modified for different lenghts)
        and C number of channels
        :param r: number of candidates to be selected randomly
        :param L_star_min: L_min = L_star_min* Q is their minimum length
        :param L_star_max: L_max = L_star_max* Q is their maximum length
        :return: candidates_notscored and random_candidates as objects from RLS_candidates class
        '''
        X = self.data
        N, Q = X.shape[0:2]

        L_max = int(L_min + step * n_steps)
    
        self.L_min = L_min
        self.L_max = L_max

        seq_random = []
        positions_random = np.array([], dtype='int')
        lengths_random = np.array([], dtype='int')
        r_size_per_length = round(r / (n_steps + 1))

        total_seq = []
        total_positions = np.array([], dtype='int')
        total_lengths = np.array([], dtype='int')
        # r_length number of random candidates for each length
        # extract r candidates subsequences equally divided in number w.r.t lengths

        total_candidates = RLS_candidateset()
        for L in range(L_min, L_max+1, step):
            # get all candidates of length L
            candidates = self.get_candidates(L)
            total_candidates.merge(candidates)

            # take random candidates per length and append everything to the random vectors
            N_candidates = len(candidates.sequences)
            random_sample = random.choice(range(N_candidates), size=r_size_per_length, replace=False)
            seq_random.extend([candidates.sequences[i] for i in random_sample])
            positions_random = np.append(positions_random, candidates.positions[random_sample])
            lengths_random = np.append(lengths_random, candidates.lengths[random_sample])

            # delete from candidates the sequences extracted randomly whose score will be computed:
            candidates = candidates.delete(random_sample)

            # append everything to the total sequences
            total_seq.extend(candidates.sequences)
            total_positions = np.append(total_positions, candidates.positions)
            total_lengths= np.append(total_lengths, candidates.lengths)

        candidates_notscored = RLS_candidateset(total_seq, total_positions, total_lengths, scores=[])
        random_candidates = RLS_candidateset(seq_random, positions_random, lengths_random, scores=[])

        # update candidates info
        self.candidates_notscored = candidates_notscored
        self.candidates_scored = random_candidates
        self.total_candidates = total_candidates
        print('the length of the total random sample should be equal to r', len(random_candidates))
        return candidates_notscored, random_candidates
        
    def compute_scores(self, candidates):
        '''
        :candidates: object from RLS_candidaset class
        return: the same object with attribute score calculated as sum of sdists
        '''
        X = self.data # train data
        N = len(X)
        r = len(candidates)
        scores = np.zeros(r)
        for i in trange(r):
            S = candidates.sequences[i]
            sum = 0
            for index in range(N):
                # sum all the sdists from every time series
                sum += util.sdist_mv(S, X[index,: ,:])
            scores[i] = sum 
        candidates.scores = scores     
        return candidates


    def compute_boundary(self, s1, sequences, distance=euclidean_distance):
        '''
        :s1: candidate sequence
        :sequences: list of numpy arrays
        :return: boundary for distances of s1 w.r.t. all the sequences
        '''
        similarity_distances = []
            # iterate over all the candidates:
        for p in range(len(sequences)):
            s2 = sequences[p]
            d = distance(mean_shift(s1), mean_shift(s2))
            # p is the index of the subsequence and d its distance to the last discovered shapelet
            similarity_distances.append(d)
        similarity_distances = np.array(similarity_distances)
        similarity_boundary = 0.1 * np.median(similarity_distances)
        return similarity_boundary


    def test_position(self, s1, pos1, best_subsequences, best_positions, similarity_boundary, pos_boundary, distance=euclidean_distance):
        '''
        test if a candidate is suitable to be added to best_subsequences, list that contains the shapelets discovered so far
        @param best_subsequences: list of numpy arrays
        @param best_positions: list of integers with same length 
        '''
        for i in range(len(best_subsequences)):
            s2 = best_subsequences[i]
            pos2 = best_positions[i]
            dist = distance(mean_shift(s1), mean_shift(s2))
            if ((abs(pos1 - pos2) < pos_boundary) or (dist < similarity_boundary)):
                return True
        return False

    def test_corr(self, s1, best_subsequences, similarity_boundary, corr_threshold, distance=euclidean_distance):
        '''
        @param s1: shapelet candidate
        @param seq_final: list  of shapelets
        @param similarity_boundary: similarity threshold calculated from 
        return: True if there exist a shapelet in seq_final with correlation >= 0.8 with the candidate
        '''
        for s2 in best_subsequences:
            corr = max_corr(s2, s1, scale='biased')
            dist = distance(mean_shift(s1), mean_shift(s2))
            if ((corr >= corr_threshold) or (dist < similarity_boundary)):
                return True
        return False
    
    def get_top_candidates(self, candidates, m, pos_boundary=0, corr_threshold=None, reverse=False, sample_size=3000):
        '''
        Extract best m best candidates in a RLS_candidateset object with computed scores.
        They must satisfy (for each i,j selected as best candidates) :
        euclidean_distance(i,j) >= similarity boundary 
        |pos_i -pos_j| >= pos_boundary
        NOTE: similarity boundary is calculated wrt the median distance to all the candidates scored
        if shapelets have different lengths, sdist is used and non euclidean distance

        :candidates_scored: RLS_candidateset object WITH NON EMPTY SCORES
        :m: number of best candidates to take
        :pos_boundary: constraint on the position of selected candidates
        :reverse: whether to select the shapelets in reverse order, aka from the one that has highest sum of distances to the lowest
        :distance: distance measure used 
        :return: RLS_candidateset object
        '''
        scores = candidates.scores
        assert scores is not None, 'Must calculate scores before'
        if len(candidates) < m:
            print('Error: too few candidates')
            return None
        
        # Set distance measure to evaluate similarity between candidates:
        if self.L_min == self.L_max:
            print('Candidates have same length')
            distance = euclidean_distance

        # if candidates have different lengths, I cannot use euclidean_distance
        else:
            distance = sdist_mv
        
        indexes = scores.argsort()
        if reverse:
            indexes = indexes[::-1]
        
        subsequences = [candidates.sequences[i] for i in indexes]

        positions = candidates.positions[indexes]
        lengths = candidates.lengths[indexes]
        scores = scores[indexes]

        # take the first one

        best_subsequences = [subsequences[0]]
        best_positions = [positions[0]]
        best_lengths = [lengths[0]]
        best_scores = [scores[0]]

        # delete it from the list of sequences

        del subsequences[0]
        positions = np.delete(positions, 0, axis=0)
        lengths = np.delete(lengths, 0, axis=0)
        scores = np.delete(scores, 0, axis=0)

        if pos_boundary is not None:
            print(f'Candidates are being filtered by a position threshold of {pos_boundary} time steps')

        elif (corr_threshold is not None):
            print(f'Candidates are being filtered by a correlation threshold of {corr_threshold}')
        
        sample_sequences_for_distance = self.total_candidates.random_sample(sample_size).sequences

        # start the candidates' selection
        while len(best_subsequences) != m:
            s1 = subsequences[0]
            pos1 = positions[0]
            score1 = scores[0]
            len1 = lengths[0]
            # calculate similarity boundary wrt all the sequences already scored
            # this prevents to have extreme cases
            similarity_boundary = self.compute_boundary(s1, sample_sequences_for_distance, distance)

            if pos_boundary is not None:
                if self.test_position(s1, pos1, best_subsequences, best_positions, similarity_boundary, pos_boundary, distance):
                    print('Distance violated')
                    del subsequences[0]
                    positions = np.delete(positions, 0, axis=0)
                    lengths = np.delete(lengths, 0, axis=0)
                    scores = np.delete(scores, 0, axis=0)
                    
                    if len(subsequences)==0:
                        break
                    continue

            if corr_threshold is not None:
                if self.test_corr(s1, best_subsequences, similarity_boundary, corr_threshold, distance):
                    del subsequences[0]
                    positions = np.delete(positions, 0, axis=0)
                    lengths = np.delete(lengths, 0, axis=0)
                    scores = np.delete(scores, 0, axis=0)
                    
                    if len(subsequences)==0:
                        break
                    continue

            print('Conditions are okkkkk')

            best_subsequences.append(s1)
            best_positions.append(pos1)
            best_scores.append(score1)
            best_lengths.append(len1)

            del subsequences[0]
            positions = np.delete(positions, 0, axis=0)
            lengths = np.delete(lengths, 0, axis=0)
            scores = np.delete(scores, 0, axis=0)

            if len(subsequences)==0:
                break 

        best_candidates = RLS_candidateset(best_subsequences, best_positions, best_lengths, best_scores)
        return best_candidates


    def LocalSearch(self, best_candidates, epsilon = (1,1), step=1, beta=0, reverse=False, maxiter=5):
        '''
        Perform a local search in candidate space, for each candidate search in self.candidates_notscored the neighbors:
        
        :param best_candidates: object from RLS_candidateset class
        :param epsilon: must be a tuple of length 2
                        - epsilon[1] is the interval of neiborhood for the position (included)
                        - epsilon[2] is the interval of neiborhood for the length (included)
        :param beta: percentage of neighbors to discard, not to compute scores
        :reverse: same meaning
        :return: RLS_candidateset object with best_candidates marged with all the neighbors found with scores computed
        '''
        candidates_notscored = self.candidates_notscored

        ### CASE WE WANT SHAPELETS IN REVERSE ORDER
        if reverse:
            for i in range(len(best_candidates)):
                print(f'Searching in candidate number {i+1} neighbors')
                j = best_candidates.positions[i]
                L = best_candidates.lengths[i]
                score = best_candidates.scores[i]
                iter=0
                
                while iter < maxiter:
                    # the method get_neighborhood eliminates N from candidates_notscored
                    N = candidates_notscored.get_neighborhood(j, L, epsilon, step, beta)

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
                    iter += 1
            return best_candidates 
        
        ### CASE WE WANT SHAPELETS TO REPRESENT NORMAL CLASS

        else:
            for i in range(len(best_candidates)):
                print(f'Searching in candidate number {i+1} neighbors')
                j = best_candidates.positions[i]
                L = best_candidates.lengths[i]
                score = best_candidates.scores[i]
                iter=0
                while iter < maxiter:
                    N = candidates_notscored.get_neighborhood(j, L, epsilon, step, beta)

                    N = self.compute_scores(N)
                    self.candidates_scored = self.candidates_scored.merge(N)
                    # min is the best score in the neighborhood if reverse is false
                    if (N.is_empty() or min(N.scores) >= score):
                        # print(f'bad search the sequences {N.sequences} have scores {N.scores}')
                        break
                    best_candidates = best_candidates.merge(N)
                    # print('neighbors found', best_candidates.sequences)

                    index = np.argwhere(N.scores ==min(N.scores))[0]
                    #update j and L to search 
                    j = N.positions[index]
                    L = N.lengths[index]
                    iter+=1
            return best_candidates  
        

    def extract(self, r, m, L_min, step=1, n_steps=0, pos_boundary=0, corr_threshold=None, epsilon=(1,1), beta=0, reverse=False, K_star = 0.02, maxiter=5, sample_size=3000):
        '''
        Extract the shapelets following RLS approach
        Update self.shapelets

        :r: initial number of random candidates
        :m: number of best candidates to take in order to look into their neighbors
        :K_star: K = K_star * Q is the number of shapelets to discover
        :L_star_min/max: same meaning
        :beta: percentage of neighbors to discard
        :return: shapelets RLS_candidateset object
        '''
        start_time = time.time()
        N, Q = self.data.shape[0:2]
        K = int(round(K_star*Q))
        if n_steps==0:
            print(f'Are going to be extracted {K} shapelets of length {L_min}')
        else:
            L_max = int(L_min + step * n_steps)
            print(f'Are going to be extracted {K} shapelets of lengths between {L_min} and {L_max}')
    
        if reverse == True :
            print('Shapelets are going to be extracted in reverse order!')
        
        _, random_candidates = self.get_random_candidates(r, L_min, step, n_steps)
        print('Finished to get random candidates')
        print('Calculating scores')
        self.compute_scores(random_candidates)
        best_candidates = self.get_top_candidates(random_candidates, m, pos_boundary, corr_threshold, reverse, sample_size)
        print(f'Finished to get top {m} candidates')
        print('Starting the local search')
        best_candidates = self.LocalSearch(best_candidates, epsilon, step, beta, reverse, maxiter)
        print('Finished local search')
        shapelets = self.get_top_candidates(best_candidates, K, pos_boundary, corr_threshold, reverse, sample_size)
        print(f'Finished to get top {K} candidates')
        self.shapelets = shapelets
        print('Time for shapelets extraction:')
        print("--- %s seconds ---" % (time.time() - start_time))
        return shapelets

    def transform(self):
        '''
        Compute the shapelet tranform of both train and test data
        :return: X_train_transform, X_test_transform numpy arrays of shape (N_train, K) and (N_test, K)
        '''
        assert self.shapelets is not None, 'Extract shapelets before'
        X_train = self.data
        X_test = self.test_data
        S = self.shapelets.sequences
        N_train = len(X_train)
        N_test = len(X_test)
        K = len(S)
        X_train_transform = np.zeros((N_train, K))
        X_test_transform = np.zeros((N_test, K))
        for k in range(K):
            shapelet = S[k]
            for i in range(N_train):
                T1 = X_train[i, :]
                d = np.sqrt(util.sdist_mv(shapelet, T1))
                X_train_transform[i, k] = d
            
            for j in range(N_test):
                T2 = X_test[j, :]
                d = np.sqrt(util.sdist_mv(shapelet, T2))
                X_test_transform[j, k] = d
        return X_train_transform, X_test_transform

# ### TEST #######

# data = np.array([[122,278,30,3], [465,45,62,4], [270,80,9,5]])
# extractor = RLS_extractor(train_data=data, test_data=None)
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