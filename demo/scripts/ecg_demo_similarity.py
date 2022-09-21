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
from src.learning.similarityloss import DiscrepancySimilarity, CorrelationSimilairty
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

# ts = X_train[y_train==1][1,:,0]
# ax[0].set_title(f'Ecg Class {1}', fontsize=10, fontweight="bold")
# ax[0].plot(ts, color='tab:blue')
# ts = X_train[y_train==0][1,:,0]
# ax[1].set_title(f'Ecg Class {-1}', fontsize=10, fontweight="bold")
# ax[1].plot(ts, color='tab:orange')
# plt.subplots_adjust(hspace=1)
# plt.savefig('ecg_classes')

np.random.seed(0)
alpha=0.01
normal_class = 1
normal_prop = 0.7
valid_prop = 0.3
X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly, X_valid_anomaly, y_valid_anomaly = preprocessing_anomaly(X_train, y_train, alpha=alpha, normal_class=normal_class, normal_prop=normal_prop, valid_prop=valid_prop)
N, Q,_ = X_train_anomaly.shape
scaler = TimeSeriesScalerMinMax()
X_train_anomaly = scaler.fit_transform(X_train_anomaly)
X_test_anomaly = scaler.fit_transform(X_test_anomaly)
X_valid_anomaly = scaler.fit_transform(X_valid_anomaly)

# plt.figure(figsize=(10, 5))
# for i in range(len(X_train_anomaly[y_train_anomaly==1][:,:,0])):
#     ts = X_train_anomaly[y_train_anomaly==1][i,:,0]
#     plt.plot(range(len(ts)), ts, c='tab:blue', label='normal' if i==0 else None)
# for i in range(len(X_train_anomaly[y_train_anomaly==-1][:,:,0])):
#     ts = X_train_anomaly[y_train_anomaly==-1][i,:,0]
#     plt.plot(range(len(ts)), ts, c='tab:orange', label='anomaly' if i==0 else None)
# plt.legend()
# plt.title('The time series in the train set', fontweight="bold")
# plt.savefig('ecg_train')

print('Ecg dataset')
print(f'number train ts is {N}, each of length {Q}')
print(f'They belong to two possible classes: {set(y_train_anomaly)}')
print('number test ts', len(X_test_anomaly))
print('number of normal train series:', len(y_train_anomaly[y_train_anomaly==1]))
print('number of anomalous train series:', len(y_train_anomaly[y_train_anomaly==-1]))
print('number of normal valid series:', len(y_valid_anomaly[y_valid_anomaly==1]))
print('number of anomalous valid series:', len(y_valid_anomaly[y_valid_anomaly==-1]))
print('number of normal test series:', len(y_test_anomaly[y_test_anomaly==1]))
print('number of anomalous test series:', len(y_test_anomaly[y_test_anomaly==-1]))


K_star =  0.2
L_star = 0.25
L = round(L_star*Q)
K = round(K_star*Q)
n_segments = N*(Q-K+1)
# clusters_centers = get_weights_via_kmeans(X_train_anomaly, len_shapelets=L, num_shapelets=K, n_segments=n_segments)
clusters_centers = np.load(f'demo/data/ecg_clusters_K={K}_L={L}.npy')
clusters_centers.shape

# np.save(f'demo/data/ecg_clusters_K={K}_L={L}', clusters_centers)



S_init = clusters_centers - np.mean(clusters_centers, axis=1, keepdims=True)
print('Type and shape of the shapelets in output', type(S_init),  S_init.shape)

plt.figure()
for i in range(len(S_init)):
    shap = S_init[i,:,]
    plt.plot(shap, label=f'shapelet{i+1}')
plt.legend()
plt.title('Ecg clusters centers', fontweight="bold")
plt.savefig('ecg_cluster_centers')

dtype = torch.double
clusters_centers = torch.tensor(clusters_centers.transpose(0,2,1), dtype=dtype)
clusters_centers = clusters_centers - torch.mean(clusters_centers, dim=2, keepdim=True)

clusters_centers = clusters_centers.double()
clusters_centers.type()
clusters_centers[0]
S_init[0] 
#### are the same!!!!!!!

print(f'The shape of cluster centers is {clusters_centers.shape}') 

# C = 1 / (N * alpha) 

C = 1 / (N * 0.1) 

print('Cuda available?', torch.cuda.is_available())
loss_sim = DiscrepancySimilarity()
extractor = LearningShapelets(len_shapelets=L, num_shapelets=K, in_channels=1, C=C, verbose=1, to_cuda=True, l1=0, loss_sim=loss_sim)
extractor.set_shapelet_weights(clusters_centers)

lr = 1e-1
optimizer = torch.optim.Adagrad(extractor.model.parameters(), lr=lr)

# lmbda = lambda epoch : 0.1
# scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
extractor.set_optimizer(optimizer)
# extractor.set_scheduler(scheduler)
n_epoch_steps = 10
n_epochs = 1
# total epochs = n_epoch_steps * n_epochs
batch_size = N

# Input shape must be a pytorch TENSOR with shape (n_samples, in_channels, len_ts)
X_train_tensor = tensor(X_train_anomaly, dtype=dtype).contiguous().transpose(1,2).cuda()
X_test_tensor = tensor(X_test_anomaly, dtype=dtype).contiguous().transpose(1,2).cuda()
X_valid_tensor = tensor(X_valid_anomaly, dtype=dtype).contiguous().transpose(1,2).cuda()

extractor.compute_radius(X_train_tensor, tol=1e-6)

extractor.model(X_train_tensor).shape

losses_dist_train = []
losses_sim = []
# losses_dist_valid = []
for epoch in range(n_epoch_steps):
    print(f'Epoch: {epoch}')
    extractor.compute_radius(X_train_tensor, tol=1e-6)
    print('radius', extractor.loss_func.get_radius())
    # current_loss_dist, current_loss_sim = extractor.fit(X_train_tensor, epochs=n_epochs, batch_size=batch_size)
    current_loss_dist = extractor.fit(X_train_tensor, epochs=n_epochs, batch_size=batch_size)
    # X_valid_transformed = extractor.model(X_valid_tensor)
    # current_loss_dist_valid = extractor.loss_func(X_valid_transformed).item()
    # print('Loss sim:', current_loss_sim)
    losses_dist_train += current_loss_dist
    # losses_dist_valid += [current_loss_dist_valid]
    # losses_sim += current_loss_sim


plt.figure()
plt.plot(losses_dist_train, color='blue')
plt.title("Loss over Training Steps", fontweight="bold")
plt.xlabel("training step")
plt.ylabel("loss")
plt.savefig('loss_ecg')


plt.figure()
plt.plot(losses_sim, color='blue')
plt.title("Loss over Training Steps", fontweight="bold")
plt.xlabel("training step")
plt.ylabel("loss")
plt.savefig('loss_ecg_sim')



X_train_transform = extractor.transform(X_train_tensor)
X_test_transform = extractor.transform(X_test_tensor)
X_valid_transform = extractor.transform(X_valid_tensor)

# check if type and shape are correct:
print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)


svdd = SVDD(C=C, zero_center=True, tol=1e-6, verbose=True, show_progress=False)

# fit the model
svdd.fit(X_train_transform)
svdd.alpha
svdd.radius

# BALANCED ACCURACY
y_test_predict = svdd.predict(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
print("Test balanced accuracy:", test_ba)
# F1 score
f1_test = f1_score(y_test_anomaly, y_test_predict, pos_label=-1)
print("Test F1 score:", f1_test)

# Confusion matrix for test 
cm = confusion_matrix(y_test_anomaly, y_test_predict)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[-1,1]).plot()
plt.savefig(f'confusion_L={L}')

# AUC
y_test_scores = -svdd.decision_function(X_test_transform)
fpr, tpr, _ = roc_curve(y_test_anomaly, y_test_scores, pos_label=-1)
auc_test = auc(fpr, tpr)
print("Test AUC:", auc_test)

y_valid_predict = svdd.predict(X_valid_transform)
valid_ba = balanced_accuracy_score(y_valid_anomaly, y_valid_predict)
print("valid balanced accuracy:", valid_ba)
# F1 score
f1_valid = f1_score(y_valid_anomaly, y_valid_predict, pos_label=-1)
print("valid F1 score:", f1_valid)

y_valid_scores = -svdd.decision_function(X_valid_transform)
fpr, tpr, _ = roc_curve(y_valid_anomaly, y_valid_scores, pos_label=-1)
auc_valid = auc(fpr, tpr)
print("Valid AUC:", auc_valid)

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
plt.savefig('ROC_ecg_K=5.png')

# OVERLAPPING HISTOGRAMS PLOT of SVDD scores (test data)

plt.figure(figsize=(8,6))
plt.hist(y_test_scores[y_test_anomaly==1], bins=40, alpha=0.5, label="Normal", color='black')
plt.hist(y_test_scores[y_test_anomaly==-1], bins=40, alpha=0.5, label="Anomalies", color='darkorange')

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histograms of SVDD scores", fontweight="bold")
plt.legend(loc='upper right')
plt.savefig('histogram_ecg_K=5')

y_train_scores = -svdd.decision_function(X_train_transform)
fpr, tpr, _ = roc_curve(y_train_anomaly, y_train_scores, pos_label=-1)
auc_train = auc(fpr, tpr)
print("Train AUC:", auc_train)

plt.figure(figsize=(8,6))
plt.hist(y_train_scores[y_train_anomaly==1], bins=400, alpha=0.5, label="Normal", color='black')
plt.hist(y_train_scores[y_train_anomaly==-1], bins=400, alpha=0.5, label="Anomalies", color='darkorange')

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histograms of SVDD scores", fontweight="bold")
plt.legend(loc='upper right')
plt.savefig('histogram_ecg_train_bad')



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
plt.savefig(f'shapelets_ecg_sim_K=1_L={L}')


### check the most useful shapelets
X_anomaly = X_test_transform[y_test_anomaly==-1]
X_normal = X_test_transform[y_test_anomaly==1]
X_anomaly = np.mean(X_anomaly, axis=0)
X_normal = np.mean(X_normal, axis=0)
diff = X_anomaly - X_normal

names=['s1', 's2', 's3']
plt.figure()
plt.bar(names, diff, color='lightcoral')
plt.xlabel('Shapelet')
plt.ylabel('Difference in mean discrepancy')
plt.savefig('difference_plot_ecg')

##### plot the best shapelet against some test series

s = S[0]
y_train_predict = svdd.predict(X_train_transform)
t = X_train_anomaly[np.logical_and(y_train_anomaly==1, y_train_predict==1)][1].flatten()
s_tensor = torch.tensor(s).reshape((1,1,L))
t_tensor = torch.tensor(t).reshape((1,1,Q))
s_tensor.shape
t_tensor.shape
num_samples, _, _ = t_tensor.shape
patches = t_tensor.unfold(dimension=1, size=1, step=1).unfold(dimension=2, size=L, step=1)
patches = patches.reshape(num_samples, -1, 1, L)
####
# added mean shift:
patches = patches - torch.mean(patches, dim=3, keepdim=True)
####
patches = torch.flatten(patches, start_dim=2, end_dim=3)
shapelets = torch.flatten(s_tensor, start_dim=1, end_dim=2)

output = torch.cdist(shapelets, patches)

# hard min compared to soft-min from the paper
output_final, pos = torch.min(output, dim=2)
pos = pos.item()
distances = output.numpy().flatten()

f, ax = plt.subplots(2, 1, sharex=True)

# pos = np.argmin(distances)
subsequence = t[pos:pos+len(s)]

s = s + np.mean(subsequence)

ax[0].plot(t)
ax[0].plot(np.arange(pos, pos + len(s)), s, linewidth=2)
ax[0].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[0].set_title("The aligned extracted shapelet", fontweight="bold")

ax[1].plot(distances)
ax[1].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[1].set_title('The discrepancies between the time series and the shapelet', fontweight="bold")

plt.tight_layout()
plt.savefig('align_ecg_normal')

