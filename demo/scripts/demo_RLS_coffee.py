import sys
sys.path.append('../')

import numpy as np
import random

from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance

from src.preprocessing_anomaly import preprocessing_anomaly
from src.SVDD.SVDD import SVDD
from src.util import Scaler, euclidean_distance_shifted, length_normalized_distance, max_corr
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

data_name = "Coffee"
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
np.random.seed(10)

# Set up anomaly detection dataset

normal_class = 0 # choose the normal class
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



# Set up the RLS extractor and hyperparameters:
# - K_star: number of shapelets in proportion to length of time series Q
# - L_min, step, n_step: range of shapelets lengths [L_min, L_min + step*n_steps]
# - r: number of candidates to be selected randomly for calculating scores at the beginning (should be not too small)
# - m: best candidates to take in order to perform LocalSearch in their neighborhood
# - epsilon: (eps[0], eps[1]): radius for neighborhood w.r.t. position, length. 
# - beta: percentage of neighbors to eliminate (not compute scores) each time a neighborhood is found 
# - maxiter: maximum number of neighborhood search for each of the m best candidates
# - reverse: bool, whether to take the shapelets with max (True) or min score
K_star = 0.02
K = round(K_star*Q)

L_star = 0.2
L = round(L_star*Q)
L_min = L
step = 1
n_steps = 0
L_max = L_min + step*n_steps
print(L_max)

r = 500
m = 10
pos_boundary = None
corr_boundary = 0.8
reverse = False
epsilon = (2,1)
beta = 0.7 # eliminate beta% of neighbors
maxiter = 4
aucs = []
scored = []
f1s = []
for seed in range(10):
    print(f'Seed: {seed}')
    np.random.seed(seed)
    extractor = RLS_extractor(train_data=X_train_anomaly, test_data=X_test_anomaly)
    print(f'Total candidates: {extractor.total_candidates}')
    # range of lengths! 
    shapelets_rls = extractor.extract(r, m, L_min, step, n_steps, pos_boundary, corr_boundary, epsilon, beta, reverse, K_star, maxiter, sample_size=r)
    S1 = shapelets_rls.sequences
    # shapelets_rls.scores
    
    print(f'Not scored: {len(extractor.candidates_notscored)}')
    print(f'Scored: {len(extractor.candidates_scored)}')
    print(f'Total: {len(extractor.total_candidates)}')
    # okkkkkkkkkkkkk
    scored.append(len(extractor.candidates_scored))
    
    channel = 0
    plt.figure()
    for i in range(len(S1)):
     shap = S1[i][:,channel]
     plt.plot(shap, label=f'shapelet{i+1}')
    plt.legend()
    plt.title('The extracted shapelets', fontweight="bold")
    plt.savefig(f'{data_name}_rls_shapelets_{seed}')
    
    
    # # test correlation SUPERGOOOOOOOOOOOOOOD
    # corr_matrix = np.zeros((K,K))
    # for i in range(len(S1)):
    #     for j in range(len(S1)):
    #         corr_matrix[i,j] = max_corr(S1[i], S1[j])
    # print(corr_matrix)
    
    
    X_train_transform, X_test_transform = extractor.transform()
    print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
    print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)
    
    
    C = 1 / (N*0.1)
    
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
    
    aucs.append(auc_test)
    # F1 score
    f1_test = f1_score(y_test_anomaly, y_test_predict, pos_label=-1)
    print("Test F1 score:", f1_test)
    f1s.append(f1_test)
np.mean(scored)
np.mean(aucs)
np.std(scored)
np.std(aucs)
np.mean(f1s)
np.std(f1s)

## check most contributing shapelets

X_anomaly = X_test_transform[y_test_anomaly==-1]
X_normal = X_test_transform[y_test_anomaly==1]
X_anomaly = np.mean(X_anomaly, axis=0)
X_normal = np.mean(X_normal, axis=0)



## plot the distance plots
#normal
f, ax = plt.subplots(2, 1, sharex=True)

t = X_test_anomaly[y_test_anomaly==1][0].flatten()

s = S1[4].flatten()
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
plt.savefig('distance')

# anomaly

f, ax = plt.subplots(2, 1, sharex=True)

t = X_test_anomaly[np.logical_and(y_test_anomaly==-1, y_test_predict==-1)][1].flatten()

s = S1[4].flatten()
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
plt.savefig('distance2')




### BRUTE FORCE

extractor_bf = Bruteforce_extractor_mv(train_data=X_train_anomaly, test_data=X_test_anomaly)
shapelets_bf = extractor_bf.extract_shapelets(K_star, L_star, pos_boundary=pos_boundary, corr_threshold=corr_boundary, reverse=reverse, sample_size=r)

S = shapelets_bf.sequences
np.save(f'results/{data_name}/s_reverse={reverse}_pos={pos_boundary}_cor={corr_boundary}_KL0.2', S)
S = np.load(f'results/{data_name}/s_reverse={reverse}_pos={pos_boundary}_cor={corr_boundary}_KL0.2.npy')
extractor_bf.shapelets = Candidateset()
extractor_bf.shapelets.sequences = S

channel = 0
plt.figure()
for i in range(K):
    shap = S[i,:,channel]
    plt.plot(shap, label=f'shapelet{i+1}')
plt.legend()
plt.title('The extracted shapelets', fontweight="bold")
plt.savefig('shapelets_coffee')

# test correlation SUPERGOOOOOOOOOOOOOOD
corr_matrix = np.zeros((K,K))
for i in range(K):
    for j in range(len(S)):
        corr_matrix[i,j] = max_corr(S[i], S[j])
print(corr_matrix)



print('Positions of extracted shapelets', shapelets_bf.positions)
print(f'Scores {shapelets_bf.scores}')
f1s = []
aus = []


extractor_bf = Bruteforce_extractor_mv(train_data=X_train_anomaly, test_data=X_test_anomaly)
extractor_bf.extract_candidates(L_star)
K = 15
shapelets = extractor_bf.get_top_candidates(K, pos_boundary, corr_boundary, reverse, sample_size=3000)
extractor_bf.shapelets = shapelets
X_train_transform, X_test_transform = extractor_bf.transform()

print('Type and shape of transformed train data', type(X_train_transform),  X_train_transform.shape)
print('Type and shape of transformed test data', type(X_test_transform),  X_test_transform.shape)

n_shap = K # n_shap must be <= K
X_train_transform = X_train_transform[:,0:n_shap]
X_test_transform = X_test_transform[:,0:n_shap]

C = 1 / (N*0.1)

svdd = SVDD(C=C, kernel='linear', zero_center=True, tol=1e-6, verbose=True)

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
aucs.append(auc_test)
f1s.append(f1_test)

aucs

# check the most contributing shapelets

X_anomaly = X_test_transform[y_test_anomaly==-1]
X_normal = X_test_transform[y_test_anomaly==1]
X_anomaly = np.mean(X_anomaly, axis=0)
X_normal = np.mean(X_normal, axis=0)
diff = X_anomaly - X_normal

names=['s1', 's2', 's3','s4', 's5', 's6']
plt.figure()
plt.bar(names, diff, color='lightcoral')
plt.xlabel('Shapelet')
plt.ylabel('Difference in mean discrepancy')
plt.savefig('difference_plot_coffee')

f, ax = plt.subplots(2, 1, sharex=True)

t = X_test_anomaly[y_test_anomaly==-1][10].flatten()

s = S[5].flatten()
distances = []
for i in range(len(t) - len(s)):
    distances.append(np.sqrt(length_normalized_distance(s,t[i:i+len(s)])))

pos = np.argmin(distances)
sequence = t[pos: pos + len(s)]

s = s - np.mean(s) + np.mean(sequence)
ax[0].plot(t)
ax[0].plot(np.arange(pos, pos + len(s)), s, linewidth=2)
ax[0].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[0].set_title("The aligned extracted shapelet", fontweight="bold")

ax[1].plot(distances)
ax[1].axvline(pos, color='k', linestyle='--', alpha=0.25)
ax[1].set_title('The discrepancies between the time series and the shapelet', fontweight="bold")

plt.tight_layout()
plt.savefig('align_coffee_anomaly')


x_max = max(X_train_transform[:,0])+0.1
y_max = max(X_train_transform[:,1])+0.1
xx, yy = np.meshgrid(np.linspace(-0.1, x_max, 500), np.linspace(-0.1, y_max, 500))
Z = svdd.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pIndex_train = y_train_anomaly == 1
nIndex_train = y_train_anomaly == -1


plt.figure()
plt.title(f"{data_name} (train)", fontweight="bold")

plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
plt.scatter(X_train_transform[pIndex_train, 0], X_train_transform[pIndex_train, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.scatter(X_train_transform[nIndex_train, 0], X_train_transform[nIndex_train, 1], facecolor='C3', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Anomalies")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                    markerscale=1.2, fancybox=True)
plt.xlim((-0.01, x_max))
plt.ylim((-0.01, y_max))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.savefig('transformed_test')



## TEST

x_max = max(X_test_transform[:,0])+0.1
y_max = max(X_test_transform[:,1])+0.1
xx, yy = np.meshgrid(np.linspace(-0.1, x_max, 500), np.linspace(-0.1, y_max, 500))
Z = svdd.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pIndex_test = y_test_anomaly == 1
nIndex_test = y_test_anomaly == -1
auc_test = np.around(auc_test, 3)

plt.figure()
plt.title(f"{data_name} (test)", fontweight="bold")

plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
plt.scatter(X_test_transform[pIndex_test, 0], X_test_transform[pIndex_test, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.scatter(X_test_transform[nIndex_test, 0], X_test_transform[nIndex_test, 1], facecolor='C3', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Anomalies")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                     markerscale=1.2, fancybox=True)

plt.xlim((-0.01, y_max))
plt.ylim((-0.01, y_max))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.savefig('transformed_test')

