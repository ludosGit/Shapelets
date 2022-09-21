import sys
sys.path.append('../')

import numpy as np
import random

from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance

from src.preprocessing_anomaly import preprocessing_anomaly
from src.SVDD.SVDD import SVDD
from src.util import Scaler, length_normalized_distance
from src.searching.bruteforce_multivariate import Bruteforce_extractor_mv, Candidateset

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

data_name = "Trace"
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

X_train = X_train[np.logical_or(y_train==3, y_train==4)]
y_train = y_train[np.logical_or(y_train==3, y_train==4)]

X_test = X_test[np.logical_or(y_test==3, y_test==4)]
y_test = y_test[np.logical_or(y_test==3, y_test==4)]
# Set up anomaly detection dataset

normal_class = 4 # choose the normal class
normal_prop = 0.8 # proportion of normal samples that go in train set
alpha = 0.1 # proportion of anomalies wrt normal 
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

# # only first channel in case multivariate
# plt.figure(figsize=(10, 5))
# for i in range(len(X_train_anomaly[y_train_anomaly==1][:,:,0])):
#     ts = X_train_anomaly[y_train_anomaly==1][i,:,0]
#     plt.plot(range(len(ts)), ts, c='tab:blue', label='normal' if i==0 else None)
# for i in range(len(X_train_anomaly[y_train_anomaly==-1][:,:,0])):
#     ts = X_train_anomaly[y_train_anomaly==-1][i,:,0]
#     plt.plot(range(len(ts)), ts, c='tab:orange', label='anomaly' if i==0 else None)
# plt.legend()
# plt.title('The time series in the train set', fontweight="bold")
# plt.savefig('trace34_train')



extractor = Bruteforce_extractor_mv(train_data=X_train_anomaly, test_data=X_test_anomaly)

K_star = 0.02 # number of shapelets in proportion to length of time series
L_star = 0.2 # length of shapelets in proportion to length of time series
L = round(L_star * Q)

# REVERSE indicates whether the extracted shapelets are the furthest (True) or the nearest (False) to the majority of the time series
reverse = True
corr_threshold = None
pos_boundary = 0
shapelets_bf = extractor.extract_shapelets(K_star, L_star, pos_boundary=pos_boundary, corr_threshold=corr_threshold, reverse=reverse, sample_size=3000)

S = np.load('results/Trace34/s_reverse=True_corr_threshold=0.8_L=55.npy')
extractor.shapelets = Candidateset()
extractor.shapelets.sequences = S

S = shapelets_bf.sequences
K = round(K_star*Q)
pos_boundary = 0
corr_threshold = None
shapelets = extractor.get_top_candidates(K, pos_boundary, corr_threshold, reverse, sample_size=3000)
S = shapelets.sequences
channel = 0
plt.figure()
for i in range(len(S)):
    shap = S[i,:,channel]
    plt.plot(shap, label=f'shapelet{i+1}')
plt.legend()
plt.title('The extracted shapelets (distance filter)', fontweight="bold")
plt.savefig('trace34_shapelets_distancefilter')


X_train_transform, X_test_transform = extractor.transform()
print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)

n_shap = 1 # n_shap must be <= K
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

x_max = max(X_train_transform[:,0])+0.05
y_max = max(X_train_transform[:,1])+0.05
pIndex_train = y_train_anomaly == 1
nIndex_train = y_train_anomaly == -1


plt.figure()
plt.title("Time Series in Transformed Space", fontweight="bold")

plt.scatter(X_train_transform[pIndex_train, 0], X_train_transform[pIndex_train, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.scatter(X_train_transform[nIndex_train, 0], X_train_transform[nIndex_train, 1], facecolor='C3', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Anomalies")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                    markerscale=1.2, fancybox=True)
plt.xlim((-0.01, x_max))
plt.ylim((-0.01, x_max))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.savefig('ex.png')



# PLOT: a test time series with the first shapelet aligned

f, ax = plt.subplots(2, 1, sharex=True)

t = X_test_anomaly[y_test_anomaly==-1][0].flatten()

s = S[0].flatten()
distances = []
for i in range(len(t) - len(s)):
    distances.append(np.sqrt(length_normalized_distance(s,t[i:i+len(s)])))

pos = np.argmin(distances)
seq = t[pos:pos+len(s)]
s = s - np.mean(s) + np.mean(seq)
ax[0].plot(t)
ax[0].plot(np.arange(pos, pos + len(s)), s, linewidth=2)
ax[0].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[0].set_title("The aligned extracted shapelet", fontweight="bold")

ax[1].plot(distances)
ax[1].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[1].set_title('The discrepancies between the time series and the shapelet', fontweight="bold")

plt.tight_layout()
plt.savefig('trace_align_anomaly')


# do the same with a normal time series

f, ax = plt.subplots(2, 1, sharex=True)

t = X_test_anomaly[y_test_anomaly==1][0].flatten()

s = S[0].flatten()
distances = []
for i in range(len(t) - len(s)):
    distances.append(np.sqrt(length_normalized_distance(s,t[i:i+len(s)])))

pos = np.argmin(distances)

ax[0].plot(t)
ax[0].plot(np.arange(pos, pos + len(s)), s, linewidth=2)
ax[0].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[0].set_title("The aligned extracted shapelet", fontweight="bold")

ax[1].plot(distances)
ax[1].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[1].set_title('The discrepancies between the time series and the shapelet', fontweight="bold")

plt.tight_layout()
plt.savefig('correct_distance2')