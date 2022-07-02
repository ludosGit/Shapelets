import sys
sys.path.append("/home/ludovicobartoli/Documents/Shapelets_first_experiments/GENDIS_python/")
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import numpy as np
from gendis.other import brute_force

try:
    from pairwise_dist import _pdist, _pdist_location
except:
    from gendis.pairwise_dist import _pdist, _pdist_location

def logloss_fitness(X, y, shapelets, cache=None, verbose=False):
    """Calculate the fitness of an individual/shapelet set
    returns the logloss score plus the sum of the lenghts to evaluate sets with same score"""
    D = np.zeros((len(X), len(shapelets)))

    # First check if we already calculated distances for a shapelet
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache_val = cache.get(shap_hash)
        if cache_val is not None:
            D[:, shap_ix] = cache_val

    # Fill up the 0 entries
    # _pdist is the pairwise distance written in C language
    _pdist(X, [shap.flatten() for shap in shapelets], D)

    # Fill up our cache
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache.set(shap_hash, D[:, shap_ix])

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(D, y)
    preds = lr.predict_proba(D)
    cv_score = -log_loss(y, preds)

    return (cv_score, sum([len(x) for x in shapelets]))


def logloss_fitness_location(X, y, shapelets, cache=None, verbose=False):
    """Calculate the fitness of an individual/shapelet set"""
    D = np.zeros((len(X), len(shapelets)))
    L = np.zeros((len(X), len(shapelets)))

    # First check if we already calculated distances for a shapelet
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache_val = cache.get(shap_hash)
        if cache_val is not None:
            D[:, shap_ix] = cache_val

    # Fill up the 0 entries
    _pdist_location(X, [shap.flatten() for shap in shapelets], D, L)

    # Fill up our cache
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache.set(shap_hash, D[:, shap_ix])

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(np.hstack((D, L)), y)
    preds = lr.predict_proba(np.hstack((D, L)))
    cv_score = -log_loss(y, preds)

    return (cv_score, sum([len(x) for x in shapelets]))


############################## TRY TO DEFINE FITNESS FOR UNSUPERVISED ANOMALY DETECTION #######################
##### COMMENTARE TUTTO IN CASO NON FUNZIONI
from trials.util import pairs_shapelet_index_gendis, permissible_error_transform, sdist_gendis

def permissible_errors_fitness(X, y, shapelets, cache=None, verbose=False):
    '''
    Compute permissible errors 
    X: time series
    shapelets: set of shapelets OF THE SAME LENGTHS
    '''
    fitness = 0
    for i in range(len(X)):
        t = X[i,:]
        pairs = pairs_shapelet_index_gendis(t, shapelets)
        distances = []
        for triplet in pairs:
            d = triplet[0]
            distances.append(d)
        fitness += -np.mean(distances)
    return (fitness, sum([len(x) for x in shapelets]))

def sdist_fitness(X, y, shapelets, cache=None, verbose=False):
    fitness = 0
    for s in shapelets: 
        for i in range(len(X)):
            t = X[i,:]
            d = sdist_gendis(s, t)
            fitness += -d
    return (fitness, sum([len(x) for x in shapelets]))

