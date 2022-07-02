import numpy as np
import sys
sys.path.append("/Users/test1/Documents/TESI_Addfor_Industriale/Python_Projects_Shapelets/shapelets_anomaly")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
## set everything bold
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
from GENDIS_python.gendis.fitness import permissible_errors_fitness, sdist_fitness
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from GENDIS_python.gendis.genetic import GeneticExtractor
from trials.preprocessing_anomaly import preprocessing_anomaly

data_name = "GunPoint"
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(data_name)

X_train = TimeSeriesScalerMinMax().fit_transform(X_train)[:,:,0]
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)[:,:,0]
print(X_train.shape)

colors = ['r', 'b', 'g', 'y', 'c']

plt.figure(1, figsize=(10, 5))
for ts, label in zip(X_train, y_train):
    plt.plot(range(len(ts)), ts, c=colors[label%len(colors)])
plt.title('The timeseries in the train set', fontweight="bold")
plt.show()

plt.figure(2, figsize=(10, 5))
for ts, label in zip(X_test, y_test):
    plt.plot(range(len(ts)), ts, c=colors[label%len(colors)])
plt.title('The timeseries in the test set', fontweight="bold")
plt.show()



## plot one sample per class

ts1 = X_train[y_train==1][0,:]
ts2 = X_train[y_train==2][0,:]
ts3 = X_train[y_train==3][0,:]
ts4 = X_train[y_train==4][0,:]

fig, ax = plt.subplots(2, 2)
ax[0][0].set_title('Trace Class 1', fontsize=10)
ax[0][0].plot(ts1)
ax[1][0].set_title('Trace Class 2', fontsize=10)
ax[1][0].plot(ts3)
ax[0][1].set_title('Trace Class 3', fontsize=10)
ax[0][1].plot(ts2)
ax[1][1].set_title('Trace Class 4', fontsize=10)
ax[1][1].plot(ts4)
plt.subplots_adjust(hspace=0.5)
plt.show()