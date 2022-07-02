import numpy as np
import sys
sys.path.append("/Users/test1/Documents/TESI_Addfor_Industriale/Python_Projects_Shapelets/Shapelets_first_experiments-search_position")
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from trials.preprocessing_anomaly import preprocessing_anomaly
from trials.RLS import RLS_candidateset, RLS_extractor
from trials.util import euclidean_distance

####################### UNIVARIATE RLS ALGORITHM TRIAL ON UCR DATASETS#################

# check if some datasets are cached 

# datasets = CachedDatasets()
# datasets.list_datasets()

# Set seed for determinism
np.random.seed(0)

data_name = "Trace"
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(data_name)


alpha = 0.05
X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly = preprocessing_anomaly(X_train, y_train, X_test, y_test, alpha, normal_class=1)

# Get statistics of the dataset
N, Q = X_train_anomaly.shape[:2]

n_classes = len(set(y_train_anomaly))
print('DATASET INFORMATION')
print(f'{data_name} dataset')
print('number train ts', N)
print('number test ts', len(X_test_anomaly))
print('number positive train', len(y_train_anomaly[y_train_anomaly==1]))
print('number positive test', len(y_test_anomaly[y_test_anomaly==1]))
print('shape of dataset loaded using tslearn', X_train_anomaly.shape)

# normalize the time series
X_train_anomaly = TimeSeriesScalerMinMax().fit_transform(X_train_anomaly)
X_test_anomaly = TimeSeriesScalerMinMax().fit_transform(X_test_anomaly)

X_train_anomaly = np.reshape(X_train_anomaly, (N, Q))
X_test_anomaly = X_test_anomaly.reshape(X_test_anomaly.shape[:2])
len(X_train_anomaly)

# SET THE PARAMETERS
extractor = RLS_extractor(train_data=X_train_anomaly, test_data=X_test_anomaly)
N, Q = extractor.data.shape[0:2]
K_star = 0.02 
K = round(K_star*Q)

# range of lengths! 

L_star_min = 45/Q
L_star_max = 55/Q
r = 500
m = 6
pos_boundary = 7
reverse = True
epsilon = (2,1)
beta = 0.7 # eliminate beta% of neighbors


# EXTRACT THE SHAPELETS
shapelets = extractor.extract(r, m, pos_boundary, epsilon, beta, reverse, \
    K_star, L_star_min=L_star_min, L_star_max=L_star_max)

shapelets.scores
shapelets.positions
shapelets.lengths
len(shapelets.sequences[0])

# plot the shapelets
extractor.plot_shapelets('shapelets.png')

# check that the sum of scored + not scored is equal to the total number of candidates

print('Number candidates scored', len(extractor.candidates_scored))
print('Number candidates not scored', len(extractor.candidates_notscored))
sum = len(extractor.candidates_scored) + len(extractor.candidates_notscored)
print('Their sum', sum)

L_min = round(Q*L_star_min)
L_max = round(Q*L_star_max)
total=0
for l in range(L_min, L_max+1):
    total+=N*(Q-l+1)

print('It should be', total)
print('They are equal:', total==sum)


############# ANOMALY DETECTION USING OCSVM ##############

# transform both train and test 
X_train_transform, X_test_transform = extractor.transform()


nu = np.around(len(y_train_anomaly[y_train_anomaly==-1])/len(y_train_anomaly), 3)
# nu = 0.045
ocsvm = OneClassSVM(nu=nu, kernel='linear')

# Choose how many extracted shapelets to take from 1 to max_n_shap
max_n_shap = round(K_star * Q)
n_shap = 2
X_train_transform = X_train_transform[:,0:n_shap]
X_test_transform = X_test_transform[:,0:n_shap]
print(X_test_transform.shape)

# fit the model
ocsvm.fit(X_train_transform)


### EVALUATE the prediction:

y_train_predict = ocsvm.predict(X_train_transform)
y_train_score = ocsvm.decision_function(X_train_transform)
train_ba = balanced_accuracy_score(y_train_anomaly, y_train_predict)
auc_train = roc_auc_score(y_train_anomaly, y_train_score)

print("OCSVM train accuracy:", train_ba)
print("OCSVM train AUC:", auc_train)

y_test_predict = ocsvm.predict(X_test_transform)
y_test_score = ocsvm.decision_function(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
auc_test = roc_auc_score(y_test_anomaly, y_test_score)

print("OCSVM test accuracy:", test_ba)
print("OCSVM test AUC:", auc_test)


################ DATA + BOUNDARY PLOTS #################

############## TRAIN DATA

x_max = max(X_train_transform[:,0])+0.01
y_max = max(X_train_transform[:,1])+0.01
xx, yy = np.meshgrid(np.linspace(-0.1, x_max, 500), np.linspace(-0.1, y_max, 500))
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pIndex_train = y_train_anomaly == 1
nIndex_train = y_train_anomaly == -1


plt.figure()
plt.title(f"{data_name}: Decision boundary (train) nu={nu}, AUC={auc_train}", fontweight="bold")

plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
plt.scatter(X_train_transform[pIndex_train, 0], X_train_transform[pIndex_train, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.scatter(X_train_transform[nIndex_train, 0], X_train_transform[nIndex_train, 1], facecolor='C3', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Anomalies")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                    markerscale=1.2, fancybox=True)
plt.xlim((-0.01, x_max ))
plt.ylim((-0.01, y_max))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
# plt.savefig(f'{data_name}_train_boundary_ocsvm_nu={nu}.png')
plt.show()

############## TEST DATA

x_max = max(X_test_transform[:,0])+0.01
y_max = max(X_test_transform[:,1])+0.01
xx, yy = np.meshgrid(np.linspace(-0.1, x_max, 500), np.linspace(-0.1, y_max, 500))
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pIndex_test = y_test_anomaly == 1
nIndex_test = y_test_anomaly == -1
auc_test = np.around(auc_test, 3)

plt.figure()
plt.title(f"Decision boundary (test) nu={nu},AUC={auc_test}", fontweight="bold")

plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
plt.scatter(X_test_transform[pIndex_test, 0], X_test_transform[pIndex_test, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.scatter(X_test_transform[nIndex_test, 0], X_test_transform[nIndex_test, 1], facecolor='C3', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Anomalies")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                     markerscale=1.2, fancybox=True)
plt.xlim((-0.01, x_max ))
plt.ylim((-0.01, y_max))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.savefig(f'{data_name}_test_boundary_ocsvm_nu={nu}.png')

