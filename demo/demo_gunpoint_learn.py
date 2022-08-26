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

data_name = "GunPoint"
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

# Set seed for determinism
np.random.seed(0)


# Set up anomaly detection dataset

normal_class = 1 # choose the normal class
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

# only first channel in case multivariate

# plt.figure(1, figsize=(10, 5))
# for ts in X_train_anomaly[y_train_anomaly==1][:,:,0]:
#     plt.plot(range(len(ts)), ts, c='tab:blue')
# for ts in X_train_anomaly[y_train_anomaly==-1][:,:,0]:
#     plt.plot(range(len(ts)), ts, c='tab:orange')
# plt.title('The time series in the train set', fontweight="bold")
# plt.savefig('gunpoint_train')

# Shapelets extracted with brute force:

extractor = Bruteforce_extractor_mv(train_data=X_train_anomaly, test_data=X_test_anomaly)

K_star = 6 / Q # number of shapelets in proportion to length of time series
L_star = 0.2 # length of shapelets in proportion to length of time series
L = round(L_star * Q)
K = round(K_star * Q)

# REVERSE indicates whether the extracted shapelets are the furthest (True) or the nearest (False) to the majority of the time series
reverse = True
corr_threshold = 0.8
pos_boundary = None

shapelets = extractor.extract_shapelets(K_star, L_star, pos_boundary=pos_boundary, corr_threshold=corr_threshold, reverse=reverse, sample_size=3000)

S = shapelets.sequences

channel = 0
plt.figure()
for i in range(K):
    shap = S[i,:,channel]
    plt.plot(shap, label=f'shapelet{i+1}')
plt.legend()
plt.title('The extracted shapelets', fontweight="bold")
plt.savefig('shapelets_gunpoint_reverse')

# transform both train and test 
X_train_transform, X_test_transform = extractor.transform()
print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)

X_anomaly = X_test_transform[y_test_anomaly==-1]
X_normal = X_test_transform[y_test_anomaly==1]
X_anomaly = np.mean(X_anomaly, axis=0)
X_normal = np.mean(X_normal, axis=0)

K = round(K_star*Q)
print('The maximum number of shapelets that can be taken is', K)
n_shap = 3 # n_shap must be <= K
X_train_transform = X_train_transform[:,0:n_shap]
X_test_transform = X_test_transform[:,0:n_shap]

ocsvm = OneClassSVM(nu=alpha, kernel='linear')

# fit the model
ocsvm.fit(X_train_transform)

# BALANCED ACCURACY
y_test_predict = ocsvm.predict(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
print("Test balanced accuracy:", test_ba)

# AUC
y_test_scores = ocsvm.decision_function(X_test_transform)
fpr, tpr, _ = roc_curve(y_test_anomaly, -y_test_scores, pos_label=-1)
auc_test = auc(fpr, tpr)
print("Test AUC:", auc_test)

# F1 score
f1_test = f1_score(y_test_anomaly, y_test_predict, pos_label=-1)
print("Test F1 score:", f1_test)


# ROC curve
fpr, tpr, _ = roc_curve(y_test_anomaly, y_test_scores)
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
plt.savefig('ROC_gunpoint.png')

# OVERLAPPING HISTOGRAMS PLOT of SVDD scores (test data)

plt.figure(figsize=(8,6))
plt.hist(y_test_scores[y_test_anomaly==1], bins=40, alpha=0.5, label="Normal", color='black')
plt.hist(y_test_scores[y_test_anomaly==-1], bins=40, alpha=0.5, label="Anomalies", color='darkorange')

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histograms of OCSVM scores", fontweight="bold")
plt.legend(loc='upper right')
plt.savefig('Gunpoint_roc')

X_test_transform = np.delete(X_test_transform, obj=1, axis=1)
X_train_transform = np.delete(X_train_transform, obj=1, axis=1)



x_max = max(X_test_transform[:,0])+0.01
y_max = max(X_test_transform[:,1])+0.01
xx, yy = np.meshgrid(np.linspace(-0.01, x_max, 500), np.linspace(-0.01, y_max, 500))
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pIndex_test = y_test_anomaly == 1
nIndex_test = y_test_anomaly == -1
auc_test = np.around(auc_test, 3)

plt.figure()
plt.title(f"{data_name} (test)", fontweight="bold")

plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="lightsteelblue")
plt.scatter(X_test_transform[pIndex_test, 0], X_test_transform[pIndex_test, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.scatter(X_test_transform[nIndex_test, 0], X_test_transform[nIndex_test, 1], facecolor='C3', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Anomalies")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                     markerscale=1.2, fancybox=True)

plt.xlim((-0.01, x_max))
plt.ylim((-0.01, y_max))
plt.ylabel("Shapelet 3")
plt.xlabel("Shapelet 1")
plt.savefig('first_two_shapelets')



K_star = 0.02
L_star = 0.2
L = round(L_star*Q)
K = round(K_star*Q)
clusters_centers = get_weights_via_kmeans(X_train_anomaly, len_shapelets=L, num_shapelets=K, n_segments=10000)
clusters_centers.shape
# # use moveaxis because the shapelets returned have shape (num_shapelets, in_channels, shapelets_size)
# S_init = np.moveaxis(clusters_centers, 1, 2)
# print('Type and shape of the shapelets in output', type(S_init),  S_init.shape)

# plt.figure()
# for i in range(len(S_init)):
#     shap = S_init[i,:,]
#     plt.plot(shap, label=f'shapelet{i+1}')
# plt.legend()
# plt.title('Random initial shapelets', fontweight="bold")
# plt.savefig('ranodm_init')

# set_shapelet_weights needs input of shape (num_shapelets, in_channels, len_shapelets)
clusters_centers = clusters_centers.transpose(0,2,1)
print(f'The shape of cluster centers is {clusters_centers.shape}') 
C = 1 / (N * alpha)
extractor = LearningShapelets(len_shapelets=L, num_shapelets=K, in_channels=n_channels, C=C, verbose=1, to_cuda=False)
extractor.set_shapelet_weights(clusters_centers)

lr = 1e-2
optimizer = torch.optim.Adagrad(extractor.model.parameters(), lr=lr)

# lmbda = lambda epoch : 0.1
# scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
extractor.set_optimizer(optimizer)
# extractor.set_scheduler(scheduler)
n_epoch_steps = 100
n_epochs = 1
# total epochs = n_epoch_steps * n_epochs
batch_size = N

# Input shape must be a pytorch TENSOR with shape (n_samples, in_channels, len_ts)
X_train_tensor = tensor(X_train_anomaly, dtype=torch.float).contiguous().transpose(1,2)

extractor.compute_radius(X_train_tensor, tol=1e-8)

losses = []
for _ in range(n_epoch_steps):
    extractor.compute_radius(X_train_tensor)
    print('radius', extractor.loss_func.get_radius())
    losses += extractor.fit(X_train_tensor, epochs=n_epochs, batch_size=batch_size)

plt.figure()
plt.plot(losses, color='blue')
plt.title("Loss over Training Steps", fontweight="bold")
plt.xlabel("training step")
plt.ylabel("loss")
plt.savefig('loss_random_init')

# tranform method takes in input a tensor of shape (n_samples, in_channels, len_ts) 
# and outputs a numpy array of shape (n_samples, n_shapelets)
X_train_transform = extractor.transform(X_train_tensor)

X_test_tensor = tensor(X_test_anomaly, dtype=torch.float).contiguous().transpose(1,2)
X_test_transform = extractor.transform(X_test_tensor)

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
plt.savefig('ROC_gunpoint_learn.png')

# OVERLAPPING HISTOGRAMS PLOT of SVDD scores (test data)

plt.figure(figsize=(8,6))
plt.hist(y_test_scores[y_test_anomaly==1], bins=3, alpha=0.5, label="Normal", color='black')
plt.hist(y_test_scores[y_test_anomaly==-1], bins=40, alpha=0.5, label="Anomalies", color='tab:orange')

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histogram of SVDD scores (test set)", fontweight="bold")
plt.legend(loc='upper right')
plt.savefig('gunpoint_histograms_learn')


y_train_scores = -svdd.decision_function(X_train_transform)
fpr, tpr, _ = roc_curve(y_train_anomaly, y_train_scores, pos_label=-1)
auc_train = auc(fpr, tpr)
print("Test AUC:", auc_train)

# ROC curve
fpr, tpr, _ = roc_curve(y_train_anomaly, y_train_scores, pos_label=-1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=3, label="ROC curve (area = %0.2f)" % 0.98)
plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.grid()
plt.savefig('ROC_gunpoint.png')

# OVERLAPPING HISTOGRAMS PLOT of SVDD scores (test data)

plt.figure(figsize=(8,6))
plt.hist(y_test_scores[y_test_anomaly==1], bins=40, alpha=0.5, label="Normal")
plt.hist(y_test_scores[y_test_anomaly==-1], bins=40, alpha=0.5, label="Anomalies")

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histograms of SVDD scores", fontweight="bold")
plt.legend(loc='upper right')
plt.savefig('Gunpoint_roc')


# F1 score
f1_test = f1_score(y_test_anomaly, y_test_predict, pos_label=-1)
print("Test F1 score:", f1_test)

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
plt.savefig('shapelets_random_init')

