import sys
sys.path.append('../')

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

data_name = "PenDigits"


X_train = np.load(f'demo/data/{data_name}_train.npy')
y_train = np.load(f'demo/data/{data_name}_train_labels.npy')
X_test = np.load(f'demo/data/{data_name}_test.npy')
y_test = np.load(f'demo/data/{data_name}_test_labeles.npy')

print(f'Shape of training data: {X_train.shape}')
print(f'Shape of test data: {X_test.shape}')
print(f'The different classes: {set(y_train)}')


X_total = np.concatenate((X_train, X_test), axis=0)
y_total = np.concatenate((y_train, y_test), axis=0)
print(f'Shape of the total data: {X_total.shape}')

#### check if one class is predominant
obs_perclass = [len(y_total[y_total==i]) for i in set(y_total)]
print(f'Number of total samples in each class: {obs_perclass}')

n_class = len(obs_perclass)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
print('The positions are the new labels', le.classes_)

scaler = TimeSeriesScalerMinMax()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_total = scaler.fit_transform(X_total)


fig, ax = plt.subplots(1, 10, figsize=(20, 2))
fig.title('Pendigits Classes', fontweight="bold")
# Plot shapelet and minimizer
for i in range(10):
    t = X_train[y_train==i][0]
    ax[i].plot(t[:, 0], t[:, 1])
    # Add an arrow
    xdata, ydata = t[:, 0], t[:, 1]
    for arw_i in range(7):
        ax[i].arrow(xdata[arw_i], ydata[arw_i], xdata[arw_i+1]-xdata[arw_i], ydata[arw_i+1]-ydata[arw_i], 
                 length_includes_head=True, head_width=0.09, color='black', overhang=0.12)

    ax[i].set_xticks([], [])
    ax[i].set_yticks([], [])
    ax[i].axis('off')
plt.savefig('pendigits_classes')


# Set seed for determinism
np.random.seed(0)


X_train = X_train[np.logical_or(y_train==5, y_train==6)]
y_train = y_train[np.logical_or(y_train==5, y_train==6)]

X_test = X_test[np.logical_or(y_test==5, y_test==6)]
y_test = y_test[np.logical_or(y_test==5, y_test==6)]

# Set up anomaly detection dataset

normal_class = 5 # choose the normal class
normal_prop = 0.8 # proportion of normal samples that go in train set
alpha = 0.05 # proportion of anomalies wrt normal 
X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly = preprocessing_anomaly(X_train, y_train, X_test, y_test, alpha, normal_class=normal_class, normal_prop=normal_prop)

N, Q, n_channels = X_train_anomaly.shape
print(f'{data_name} dataset')
print(f'number train ts is {N}, each of length {Q}')
print(f'They belong to two possible classes: {set(y_train_anomaly)}')
print('number test ts', len(X_test_anomaly))
print('number of normal train series:', len(y_train_anomaly[y_train_anomaly==1]))
print('number of anomalous train series:', len(y_train_anomaly[y_train_anomaly==-1]))
print('number of normal test series:', len(y_test_anomaly[y_test_anomaly==1]))
print('number of anomalous test series:', len(y_test_anomaly[y_test_anomaly==-1]))

scaler = TimeSeriesScalerMinMax()
X_train_anomaly = scaler.fit_transform(X_train_anomaly)
X_test_anomaly = scaler.transform(X_test_anomaly)

## check it by printing some time series

t = X_train_anomaly[y_train_anomaly==1][60]
plt.figure()
plt.plot(t[:, 0], t[:, 1], label='Learnt shapelet')
plt.savefig('troia')


# only first channel in case multivariate

# plt.figure(1, figsize=(10, 5))
# for ts in X_train_anomaly[y_train_anomaly==1][:,:,0]:
#     plt.plot(range(len(ts)), ts, c='tab:blue')
# for ts in X_train_anomaly[y_train_anomaly==-1][:,:,0]:
#     plt.plot(range(len(ts)), ts, c='tab:orange')
# plt.title('The time series in the train set', fontweight="bold")
# plt.savefig('gunpoint_train')



K_star = 1/Q
L_star = 4/Q
L = round(L_star*Q)
K = round(K_star*Q)
# clusters_centers = get_weights_via_kmeans(X_train_anomaly, len_shapelets=L, num_shapelets=K, n_segments=10000)
# clusters_centers.shape
# clusters_centers = clusters_centers.transpose(0,2,1)

# # use moveaxis because the shapelets returned have shape (num_shapelets, in_channels, shapelets_size)
# S_init = np.moveaxis(clusters_centers, 1, 2)
# print('Type and shape of the shapelets in output', type(S_init),  S_init.shape)

# plt.figure()
# for i in range(len(S_init)):
#     shap = S_init[i,:,]
#     plt.plot(shap[:,0], shap[:,1], label=f'shapelet{i+1}')
# plt.legend()
# plt.title('Random initial shapelets', fontweight="bold")
# plt.savefig('clusters')

# # set_shapelet_weights needs input of shape (num_shapelets, in_channels, len_shapelets)
# print(f'The shape of cluster centers is {clusters_centers.shape}') 
C = 1 / (N * alpha)
extractor = LearningShapelets(len_shapelets=L, num_shapelets=K, in_channels=n_channels, C=C, verbose=1, to_cuda=True)
# extractor.set_shapelet_weights(clusters_centers)

lr = 1e-1
optimizer = torch.optim.Adagrad(extractor.model.parameters(), lr=lr)

# lmbda = lambda epoch : 0.1
# scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
extractor.set_optimizer(optimizer)
# extractor.set_scheduler(scheduler)
n_epoch_steps = 400
n_epochs = 1
# total epochs = n_epoch_steps * n_epochs
batch_size = N

# Input shape must be a pytorch TENSOR with shape (n_samples, in_channels, len_ts)
X_train_tensor = tensor(X_train_anomaly, dtype=torch.float).contiguous().transpose(1,2)

X = extractor.transform(X_train_tensor).reshape((N,1)) # numpy array shape (n_samples, n_shapelets)

svdd = SVDD(C=C, zero_center=True, verbose=False, tol=1e-8)
svdd.fit(X)
extractor.loss_func.update_r(svdd.radius)


losses = []
for _ in range(n_epoch_steps):
    X = extractor.transform(X_train_tensor).reshape((N,1)) # numpy array shape (n_samples, n_shapelets)
    
    svdd = SVDD(C=C, zero_center=True, verbose=False, tol=1e-8)
    svdd.fit(X)
    extractor.loss_func.update_r(svdd.radius)
    print('radius', extractor.loss_func.get_radius())
    losses += extractor.fit(X_train_tensor, epochs=n_epochs, batch_size=batch_size)

plt.figure()
plt.plot(losses, color='blue')
plt.title("Loss over Training Steps", fontweight="bold")
plt.xlabel("training step")
plt.ylabel("loss")
plt.savefig('loss')

# tranform method takes in input a tensor of shape (n_samples, in_channels, len_ts) 
# and outputs a numpy array of shape (n_samples, n_shapelets)
X_train_transform = extractor.transform(X_train_tensor).reshape((N,1))

X_test_tensor = tensor(X_test_anomaly, dtype=torch.float).contiguous().transpose(1,2)
X_test_transform = extractor.transform(X_test_tensor).reshape((len(X_test_anomaly),1))

# check if type and shape are correct:
print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)

svdd = SVDD(C=C, zero_center=True)

# fit the model
svdd.fit(X_train_transform)

# BALANCED ACCURACY
y_test_predict = svdd.predict(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
print("Test balanced accuracy:", test_ba)

# AUC
y_test_scores = -svdd.decision_function(X_test_transform)
fpr, tpr, _ = roc_curve(y_test_anomaly, y_test_scores, pos_label=-1)
auc_test = auc(fpr, tpr)
print("Test AUC:", auc_test)

# F1 score
f1_test = f1_score(y_test_anomaly, y_test_predict, pos_label=-1)
print("Test F1 score:", f1_test)

S = extractor.get_shapelets()
# use moveaxis because the shapelets returned have shape (num_shapelets, in_channels, shapelets_size)
S = np.moveaxis(S, 1, 2)

plt.figure()
for i in range(len(S)):
    t = X_train_anomaly[y_train_anomaly==1][67]
    plt.plot(t[:,0], t[:,1], label='time series')
    xdata, ydata = t[:, 0], t[:, 1]
    for arw_i in range(7):
        plt.arrow(xdata[arw_i], ydata[arw_i], xdata[arw_i+1]-xdata[arw_i], ydata[arw_i+1]-ydata[arw_i], 
                 length_includes_head=True, head_width=0.02, color='tab:blue', overhang=0.04)
    shap = S[i,:,]
    plt.plot(shap[:,0], shap[:,1], label=f'shapelet')
    xdata, ydata = shap[:, 0], shap[:, 1]
    for arw_i in range(3):
        plt.arrow(xdata[arw_i], ydata[arw_i], xdata[arw_i+1]-xdata[arw_i], ydata[arw_i+1]-ydata[arw_i], 
                 length_includes_head=True, head_width=0.02, color='tab:orange', overhang=0.04)
plt.legend()
plt.title('Normal test time series', fontweight="bold")
plt.savefig('pendigits_normal')

plt.figure()
for i in range(len(S)):
    t = X_test_anomaly[y_test_anomaly==-1][6]
    plt.plot(t[:,0], t[:,1], label='time series')
    xdata, ydata = t[:, 0], t[:, 1]
    for arw_i in range(7):
        plt.arrow(xdata[arw_i], ydata[arw_i], xdata[arw_i+1]-xdata[arw_i], ydata[arw_i+1]-ydata[arw_i], 
                 length_includes_head=True, head_width=0.02, color='tab:blue', overhang=0.04)
    shap = S[i,:,]
    plt.plot(shap[:,0], shap[:,1], label='shapelet')
    xdata, ydata = shap[:, 0], shap[:, 1]
    for arw_i in range(3):
        plt.arrow(xdata[arw_i], ydata[arw_i], xdata[arw_i+1]-xdata[arw_i], ydata[arw_i+1]-ydata[arw_i], 
                 length_includes_head=True, head_width=0.02, color='tab:orange', overhang=0.04)
plt.legend()
plt.title('Anomalous test time series', fontweight="bold")
plt.savefig('pendigits_anomaly')