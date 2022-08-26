import pandas as pd

import sys
sys.path.append('../')

import os
print(os.getcwd())

import numpy as np
import random
import torch
from torch import tensor

from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance

from src.preprocessing_anomaly import preprocessing_anomaly
from src.SVDD.SVDD import SVDD
from src.util import Scaler, euclidean_distance_shifted, length_normalized_distance, max_corr, get_weights_via_kmeans
from src.learning.learningextractor import LearningShapelets 
from src.searching.bruteforce_multivariate import Bruteforce_extractor_mv, Candidateset
from src.searching.RLS import RLS_extractor, RLS_candidateset

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


data_name = 'ecg'

dataframe = pd.read_csv('demo/data/ecg.csv', header=None)
raw_data = dataframe.values
X_train = dataframe.to_numpy()
print('Data shape', X_train.shape)
y_train = np.array(X_train[:,140], dtype='int')
X_train = np.delete(X_train, obj=140, axis=1)
N, Q = X_train.shape
X_train = X_train.reshape((N,Q,1))

len(y_train[y_train==0])
obs_perclass = [len(y_train[y_train==i]) for i in set(y_train)]
print(f'Number of total samples in each class: {obs_perclass}')
n_class = len(obs_perclass)

# fig, ax = plt.subplots(n_class,)

# for i in range(n_class):
#     ts = X_train[y_train==i][0,:,0]
#     ax[i].set_title(f'Ecg Class {i+1}', fontsize=10, fontweight="bold")
#     ax[i].plot(ts)
# plt.subplots_adjust(hspace=1)
# plt.savefig('ecg_classes')

alpha=0.001
normal_class = 0
normal_prop = 0.8
X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly = preprocessing_anomaly(X_train, y_train, alpha=alpha, normal_class=normal_class, normal_prop=normal_prop, valid_prop=0)
N, Q,_ = X_train_anomaly.shape
print('Ecg dataset')
print(f'number train ts is {N}, each of length {Q}')
print(f'They belong to two possible classes: {set(y_train_anomaly)}')
print('number test ts', len(X_test_anomaly))
print('number of normal train series:', len(y_train_anomaly[y_train_anomaly==1]))
print('number of anomalous train series:', len(y_train_anomaly[y_train_anomaly==-1]))
print('number of normal test series:', len(y_test_anomaly[y_test_anomaly==1]))
print('number of anomalous test series:', len(y_test_anomaly[y_test_anomaly==-1]))


# plt.figure(1, figsize=(10, 5))
# for ts in X_train_anomaly[y_train_anomaly==1][:,:,0]:
#     plt.plot(range(len(ts)), ts, c='tab:blue')
# for ts in X_train_anomaly[y_train_anomaly==-1][:,:,0]:
#     plt.plot(range(len(ts)), ts, c='tab:orange')
# plt.title('The time series in the train set', fontweight="bold")
# plt.savefig('ecg_train')

K_star = 6/Q
L_star = 0.3
L = round(L_star*Q)
K = round(K_star*Q)

C = 1 / (N * alpha)

extractor = RLS_extractor(train_data=X_train_anomaly, test_data=X_test_anomaly)
N, Q, n_channels = extractor.data.shape


L_min = L
step = 1
n_steps = 0
L_max = L_min + step*n_steps
print(L_max)

r = 500
m = K
pos_boundary = None
corr_boundary = 0.8
reverse = False
epsilon = (2,0)
beta = 0.7 # eliminate beta% of neighbors
maxiter = 4

shapelets_rls = extractor.extract(r, m, L_min, step, n_steps, pos_boundary, corr_boundary, epsilon, beta, reverse, K_star, maxiter, sample_size=r)
S1 = shapelets_rls.sequences
shapelets_rls.scores

len(extractor.candidates_notscored)

channel = 0
plt.figure()
for i in range(len(S1)):
    shap = S1[i][:,channel]
    plt.plot(shap, label=f'shapelet{i+1}')
plt.legend()
plt.title('The extracted shapelets', fontweight="bold")
plt.savefig(f'{data_name}_rls_shapelets1')


# test correlation SUPERGOOOOOOOOOOOOOOD
corr_matrix = np.zeros((K,K))
for i in range(len(S1)):
    for j in range(len(S1)):
        corr_matrix[i,j] = max_corr(S1[i], S1[j])
print(corr_matrix)

print('Number candidates scored.', len(extractor.candidates_scored))
print('Number candidates not scored.', len(extractor.candidates_notscored))
sum = len(extractor.candidates_scored) + len(extractor.candidates_notscored)
print('Their sum', sum)

total=0
for l in range(L_min, L_max+1, step):
    total+=N*(Q-l+1)

print('It should be', total)

print('Positions of extracted shapelets', shapelets_rls.positions)
print(f'Scores {shapelets_rls.scores}')


X_train_transform, X_test_transform = extractor.transform()
print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)

n_shap = K # n_shap must be <= K
X_train_transform = X_train_transform[:,0:n_shap]
X_test_transform = X_test_transform[:,0:n_shap]

C = 1 / (N*alpha)

svdd = SVDD(C=C, kernel='linear', zero_center=True, tol=1e-6, verbose=False)

# fit the model
svdd.fit(X_train_transform)

# BALANCED ACCURACY
y_test_predict = svdd.predict(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
print("Test balanced accuracy:", test_ba)

# AUC
y_test_scores = svdd.decision_function(X_test_transform)
auc_test = roc_auc_score(y_test_anomaly, y_test_scores)
print("Test AUC:", auc_test)

# F1 score
f1_test = f1_score(y_test_anomaly, y_test_predict, pos_label=-1)
print("Test F1 score:", f1_test)