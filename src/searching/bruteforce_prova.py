import numpy as np
import sys
sys.path.append("/Documents/Shapelets_first_experiments")
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.preprocessing import MinMaxScaler

from src.searching.bruteforce import Bruteforce_extractor
from src.preprocessing_anomaly import preprocessing_anomaly
from src.util import euclidean_distance

####################### UNIVARIATE BRUTEFORCE ALGORITHM TRIAL ON UCR DATASETS #################

# Set seed for determinism
np.random.seed(0)

# Download data
data_name = "Trace"
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(data_name)

######################### PREPROCESSING #########################

# Set up anomaly detection dataset

alpha = 0.05 # percentage of anomalies
X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly = preprocessing_anomaly(X_train, y_train, X_test, y_test, alpha, normal_class=1)

# Get statistics of the dataset
N, Q = X_train_anomaly.shape[:2]
n_classes = len(set(y_train_anomaly))
print(f'{data_name} dataset')
print('number train ts', N)
print('number test ts', len(X_test_anomaly))
print('number positive train', len(y_train_anomaly[y_train_anomaly==1]))
print('number positive test', len(y_test_anomaly[y_test_anomaly==1]))

shape = X_train_anomaly.shape
X_train_anomaly = X_train_anomaly.flatten()
# normalize the time series
# as in the tutorial https://tslearn.readthedocs.io/en/stable/auto_examples/classification/plot_shapelets.html#sphx-glr-auto-examples-classification-plot-shapelets-py
X_train_anomaly = MinMaxScaler().fit_transform(X_train_anomaly.reshape(np.product(shape), 1))
X_train_anomaly  = X_train_anomaly.reshape(shape)
X_train_anomaly.shape

X_test_anomaly = TimeSeriesScalerMinMax().fit_transform(X_test_anomaly)

# reshape
X_train_anomaly = X_train_anomaly.reshape((N,Q))
X_test_anomaly = X_test_anomaly.reshape(X_test_anomaly.shape[:2])

# Brute force extraction:

extractor = Bruteforce_extractor(train_data=X_train_anomaly, test_data=X_test_anomaly)
K_star = 0.02
L_star = 0.2
# reverse indicates whether the extracted shapelets are the furthest or the nearest to the majority of the time series
reverse = True
shapelets = extractor.extract_shapelets(K_star, L_star, pos_boundary=0, reverse=reverse)
S = shapelets.sequences

extractor.plot_shapelets('shap.png')
shapelets.positions



############# ANOMALY DETECTION USING OCSVM ##############

# transform both train and test 
X_train_transform, X_test_transform = extractor.transform()

nu = 0.045
ocsvm = OneClassSVM(nu=nu, kernel='linear')

# Choose how many extracted shapelets to take from 1 to max_n_shap
max_n_shap = round(K_star * Q)
n_shap = 2
X_train_transform = X_train_transform[:,0:n_shap]
X_test_transform = X_test_transform[:,0:n_shap]
print(X_train_transform.shape)

# fit the model
ocsvm.fit(X_train_transform)

### EVALUATE the prediction:

y_train_predict = ocsvm.predict(X_train_transform)
y_train_score = ocsvm.score_samples(X_train_transform)
train_ba = balanced_accuracy_score(y_train_anomaly, y_train_predict)
auc_train = roc_auc_score(y_train_anomaly, y_train_score)
print("OCSVM train accuracy:", train_ba)
print("OCSVM train AUC:", auc_train)

y_test_predict = ocsvm.predict(X_test_transform)
y_test_score = ocsvm.score_samples(X_test_transform)
test_ba = balanced_accuracy_score(y_test_anomaly, y_test_predict)
auc_test = roc_auc_score(y_test_anomaly, y_test_score)
print("OCSVM test accuracy:", test_ba)
print("OCSVM test AUC:", auc_test)

################ PLOTS #################


