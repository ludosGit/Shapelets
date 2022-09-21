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
from src.util import Scaler, euclidean_distance_shifted, length_normalized_distance, max_corr, get_weights_via_kmeans, mean_shift
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
indices = [366, 426]
X_train_anomaly = np.delete(X_train_anomaly, indices)

K_star = 6/Q
L_star = 0.3
L = round(L_star*Q)
K = round(K_star*Q)
n_segments = N*(Q-K+1)
clusters_centers = get_weights_via_kmeans(X_train_anomaly, len_shapelets=L, num_shapelets=K, n_segments=n_segments)
clusters_centers.shape



S_init = clusters_centers - np.mean(clusters_centers, axis=1, keepdims=True)
print('Type and shape of the shapelets in output', type(S_init),  S_init.shape)

plt.figure()
for i in range(len(S_init)):
    shap = S_init[i,:,]
    plt.plot(shap, label=f'shapelet{i+1}')
plt.legend()
plt.title('Random initial shapelets', fontweight="bold")
plt.savefig('ecg_cluster_centers')

clusters_centers = torch.tensor(clusters_centers.transpose(0,2,1))
clusters_centers = clusters_centers - torch.mean(clusters_centers, dim=2, keepdim=True)

clusters_centers = clusters_centers.double()
clusters_centers.type()
clusters_centers[0]
S_init[0] 
#### are the same!!!!!!!

print(f'The shape of cluster centers is {clusters_centers.shape}') 
C = 1 / (N * alpha)

print('Cuda available?', torch.cuda.is_available())
extractor = LearningShapelets(len_shapelets=L, num_shapelets=K, in_channels=1, C=C, verbose=1, to_cuda=True)
extractor.set_shapelet_weights(clusters_centers)

lr = 1e-2
optimizer = torch.optim.Adagrad(extractor.model.parameters(), lr=lr)

# lmbda = lambda epoch : 0.1
# scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
extractor.set_optimizer(optimizer)
# extractor.set_scheduler(scheduler)
n_epoch_steps = 200
n_epochs = 1
# total epochs = n_epoch_steps * n_epochs
batch_size = N

# Input shape must be a pytorch TENSOR with shape (n_samples, in_channels, len_ts)
X_train_tensor = tensor(X_train_anomaly, dtype=torch.double).contiguous().transpose(1,2)

extractor.compute_radius(X_train_tensor, tol=1e-9)

losses = []
for _ in range(n_epoch_steps):
    extractor.compute_radius(X_train_tensor, tol=1e-9)
    print('radius', extractor.loss_func.get_radius())
    losses += extractor.fit(X_train_tensor, epochs=n_epochs, batch_size=batch_size)

plt.figure()
plt.plot(losses, color='blue')
plt.title("Loss over Training Steps", fontweight="bold")
plt.xlabel("training step")
plt.ylabel("loss")
plt.savefig('loss_ecg')

X_train_transform = extractor.transform(X_train_tensor)

X_test_tensor = tensor(X_test_anomaly, dtype=torch.double).contiguous().transpose(1,2)
X_test_transform = extractor.transform(X_test_tensor)

# check if type and shape are correct:
print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)

svdd = SVDD(C=C, zero_center=True, tol=1e-9)
C = 0.001
# fit the model
svdd.fit(X_train_transform)
# print(max(svdd.alpha))
# svdd.boundary_sv_index
# svdd.decision_function(X_train_transform[366].reshape(-1,6))

# plt.figure()
# plt.plot(X_train_anomaly[366])
# plt.plot(X_train_anomaly[367])
# plt.plot(X_train_anomaly[y_train_anomaly==-1][0])
# plt.savefig('bastarda')
# y_train_anomaly[367]

# BALANCED ACCURACY
y_test_predict = svdd.predict(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
print("Test balanced accuracy:", test_ba)

# Confusion matrix for test 
cm = confusion_matrix(y_test_anomaly, y_test_predict)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.savefig('confusion')

# AUC
y_test_scores = -svdd.decision_function(X_test_transform)
fpr, tpr, _ = roc_curve(y_test_anomaly, y_test_scores, pos_label=-1)
auc_test = auc(fpr, tpr)
print("Test AUC:", auc_test)

# ROC curve
fpr, tpr, _ = roc_curve(y_test_anomaly, y_test_scores, pos_label=-1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=3, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.grid()
plt.savefig('ROC_ecg.png')

# OVERLAPPING HISTOGRAMS PLOT of SVDD scores (test data)

plt.figure(figsize=(8,6))
plt.hist(y_test_scores[y_test_anomaly==1], bins=40, alpha=0.5, label="Normal", color='black')
plt.hist(y_test_scores[y_test_anomaly==-1], bins=40, alpha=0.5, label="Anomalies", color='darkorange')

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histograms of SVDD scores", fontweight="bold")
plt.legend(loc='upper right')
plt.savefig('histogram_ecg')

y_train_scores = -svdd.decision_function(X_train_transform)
fpr, tpr, _ = roc_curve(y_train_anomaly, y_train_scores, pos_label=-1)
auc_train = auc(fpr, tpr)
print("Test AUC:", auc_train)

plt.figure(figsize=(8,6))
plt.hist(y_train_scores[y_train_anomaly==1], bins=400, alpha=0.5, label="Normal", color='black')
plt.hist(y_train_scores[y_train_anomaly==-1], bins=400, alpha=0.5, label="Anomalies", color='darkorange')

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histograms of SVDD scores", fontweight="bold")
plt.legend(loc='upper right')
plt.savefig('histogram_ecg_train')



# plot the shapelets
S = extractor.get_shapelets()
# use moveaxis because the shapelets returned have shape (num_shapelets, in_channels, shapelets_size)
S = np.moveaxis(S, 1, 2)
print('Type and shape of the shapelets in output', type(S),  S.shape)

plt.figure()
for i in range(len(S)):
    shap = S[i,:,]
    plt.plot(shap, label=f'shapelet{i+1}')
plt.legend()
plt.title('The extracted shapelets', fontweight="bold")
plt.savefig('shapelets_ecg')