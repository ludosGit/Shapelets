import numpy as np
import sys
sys.path.append("/Documents/Shapelets_first_experiments")
from sklearn.svm import OneClassSVM
from src.SVDD.SVDD import SVDD
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from src.searching.bruteforce_multivariate import Bruteforce_extractor_mv, Candidateset
from src.preprocessing_anomaly import preprocessing_anomaly
from src.util import Normalizer

####################### MULTIVARIATE BRUTEFORCE ALGORITHM TRIAL ON UCR DATASETS#################

#### Check chached datsets
datasets = CachedDatasets()
datasets.list_datasets()

# Set seed for determinism
np.random.seed(0)

# Download data
data_name = "BasicMotions"
# X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(data_name)
# np.save(f'demo/data/{data_name}_train', X_train)
# np.save(f'demo/data/{data_name}_train_labels', y_train)
# np.save(f'demo/data/{data_name}_test', X_test)
# np.save(f'demo/data/{data_name}_test_labeles', y_test)

X_train = np.load(f'demo/data/{data_name}_train.npy')
y_train = np.load(f'demo/data/{data_name}_train_labels.npy')
X_test = np.load(f'demo/data/{data_name}_test.npy')
y_test = np.load(f'demo/data/{data_name}_test_labeles.npy')

######################### PREPROCESSING #########################

print(X_train.shape)
print(X_test.shape)
print(set(y_train))
le = LabelEncoder()
le.classes_
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
X_total = np.concatenate((X_train, X_test), axis=0)
y_total = np.concatenate((y_train, y_test), axis=0)
X_total.shape

#### check if one class is predominant
n_class = [len(y_total[y_total==i]) for i in set(y_total)]

# Set up anomaly detection dataset
alpha = 0.1 # percentage of anomalies

X_train_anomaly, y_train_anomaly, \
     X_test_anomaly, y_test_anomaly = preprocessing_anomaly(X_train, y_train, X_test, 
     y_test, alpha, normal_class=1)

N, Q = X_train_anomaly.shape[:2]
n_classes = len(set(y_train_anomaly))
print(f'{data_name} dataset')
print('number train ts', N)
print('number test ts', len(X_test_anomaly))
print('number positive train', len(y_train_anomaly[y_train_anomaly==1]))
print('number positive test', len(y_test_anomaly[y_test_anomaly==1]))


# NORMALIZE the time series
# MINMAX: X - min / max - min; applied to each time series with local min, max !!!!!!
# # Examples:
# # univariate
# x = np.array([[1,2,3], [4,5,6], [9, 10, 11]])
# x = TimeSeriesScalerMinMax().fit_transform(x) # all the same!!

# # multivariate
# x = [np.array([[1,2,3,4,5], [6,7,8,9,10]]), np.array([[34,2,23,4,5], [6,72,81,9,10]])]
# x = np.array(x)
# x = np.moveaxis(x, -1, 1)
# x = TimeSeriesScalerMinMax().fit_transform(x)
# # in multivariate case it is done by channel!!!

# ### can be used also meanvariance
# X_train_anomaly = TimeSeriesScalerMeanVariance().fit_transform(X_train_anomaly)
# X_test_anomaly = TimeSeriesScalerMeanVariance().fit_transform(X_test_anomaly)

# X_train_anomaly = TimeSeriesScalerMinMax().fit_transform(X_train_anomaly)
# X_test_anomaly = TimeSeriesScalerMinMax().fit_transform(X_test_anomaly)

normalizer = Normalizer(scaler=MinMaxScaler())
X_train_anomaly = normalizer.fit_normalize(X_train_anomaly)
X_test_anomaly = normalizer.normalize(X_test_anomaly)

X_train_anomaly.shape
extractor = Bruteforce_extractor_mv(train_data=X_train_anomaly, test_data=X_test_anomaly)
K_star = 0.02
L_star = 0.2

# REVERSE indicates wheter the extracted shapelets are the furthest (True) or the nearest (False) to the majority of the time series
reverse = False
pos_boundary=5
shapelets = extractor.extract_shapelets(K_star, L_star, pos_boundary=pos_boundary, reverse=reverse)
S = shapelets.sequences
np.save(f'results/{data_name}/s_reverse={reverse}_pos={pos_boundary}.npy', S)
shapelets.positions
shapelets.scores

############# ANOMALY DETECTION USING SVDD ##############

# S = np.load(f'results/{data_name}/s_reverse={reverse}_pos={pos_boundary}.npy')
# # set manually the shapelets in extractor
# extractor.shapelets = Candidateset()
# extractor.shapelets.sequences = S
# transform both train and test 
X_train_transform, X_test_transform = extractor.transform()
X_train_transform.shape

# nu is the proportion of anoamlies w.r.t. the total
nu = np.around(len(y_train_anomaly[y_train_anomaly==-1])/len(y_train_anomaly), 3)

# recompute alpha because the approximation is too big
# round(16*0.1)
# np.around(2/16, 3)
alpha = np.around(len(y_train_anomaly[y_train_anomaly==-1])/len(y_train_anomaly[y_train_anomaly==1]), 3)

# set C with real percentage of anomalies
C = 1 / (N * nu)

# set C according to Beggel's paper (smaller)
C = 1 / (N * alpha)



svdd = SVDD(C=C, zero_center=True)

# Choose how many extracted shapelets to take from 1 to max_n_shap
max_n_shap = round(K_star * Q)
n_shap = 2
X_train_transform = X_train_transform[:,0:n_shap]
X_test_transform = X_test_transform[:,0:n_shap]
print(X_train_transform.shape)

# fit the model
svdd.fit(X_train_transform)
svdd.radius
svdd.boundary_sv_index

########## EVALUATE the prediction:

## TRAIN
# BALANCED ACCURACY
y_train_predict = svdd.predict(X_train_transform)
train_ba = balanced_accuracy_score(y_train_anomaly, y_train_predict)
print("SVDD train accuracy:", train_ba)

# AUC
y_train_scores = svdd.decision_function(X_train_transform)
auc_train = roc_auc_score(y_train_anomaly, y_train_scores)
print("SVDD train AUC:", auc_train)

## TEST
# BALANCED ACCURACY
y_test_predict = svdd.predict(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
print("SVDD test accuracy:", test_ba)

# AUC
y_test_scores = svdd.decision_function(X_test_transform)
auc_test = roc_auc_score(y_test_anomaly, y_test_scores)
print("SVDD test AUC:", auc_test)

# Confusion matrix
cm = confusion_matrix(y_test_anomaly, y_test_predict)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
# plt.savefig('confusion.png')

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
plt.show()
# plt.savefig('ROC_test.png')

# OVERLAPPING HISTOGRAMS PLOT of SVDD scores (test data)

plt.figure(figsize=(8,6))
plt.hist(y_test_scores[y_test_anomaly==1], bins=40, alpha=0.5, label="Normal")
plt.hist(y_test_scores[y_test_anomaly==-1], bins=40, alpha=0.5, label="Anomalies")

plt.xlabel("Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Histograms of SVDD scores", fontweight="bold")
plt.legend(loc='upper right')
plt.show()
# plt.savefig("hist.png")


################ DATA + BOUNDARY PLOTS #################

############## TRAIN DATA

x_max = max(X_train_transform[:,0])+0.01
y_max = max(X_train_transform[:,1])+0.01
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
plt.xlim((-0.01, x_max + 0.1))
plt.ylim((-0.01, y_max + 0.1))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.show()
# plt.savefig(f'Train_boundary_svdd_ba={train_ba}.png')

############## TEST DATA

x_max = max(X_test_transform[:,0])+0.01
y_max = max(X_test_transform[:,1])+0.01
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

plt.xlim((-0.01, x_max + 0.1))
plt.ylim((-0.01, y_max + 0.1))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.show()
# plt.savefig(f'{data_name}_test_boundary_ocsvm_nu={nu}.png')











