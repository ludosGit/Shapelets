import numpy as np
import sys
sys.path.append("/home/ludovicobartoli/Documents/Shapelets_first_experiments")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from GENDIS_python.gendis.fitness import permissible_errors_fitness, sdist_fitness
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from GENDIS_python.gendis.genetic import GeneticExtractor
from trials.preprocessing_anomaly import preprocessing_anomaly

print('END IMPORT')

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("GunPoint")
alpha = 0.1
X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly = preprocessing_anomaly(X_train, y_train, X_test, y_test, alpha, normal_class=1)

X_train_anomaly = TimeSeriesScalerMinMax().fit_transform(X_train_anomaly)[:,:,0]
X_test_anomaly = TimeSeriesScalerMinMax().fit_transform(X_test_anomaly)[:,:,0]
L = 15

print('DATA LOADTING AND SPLITTING')
genetic_extractor = GeneticExtractor(population_size=5, iterations=5, verbose=True,
                                     mutation_prob=0.3, crossover_prob=0.3,
                                     wait=10, max_len=L, fitness=permissible_errors_fitness)
                                     
print('INSTANTIATED GENETIC EXTRACTOR')
### NOTE: when extracting shapelets, 'avg' is the average fitness value, which INCREASES as more populations are generated

print('START GENETIC EXTRACTOR FIT')
shapelets = genetic_extractor.fit(X_train_anomaly, y_train_anomaly)

## ONE SHAPELET extracted

plt.figure()
for shap in genetic_extractor.shapelets:
    plt.plot(shap)
plt.title('Extracted shapelets - ECGFiveDays')
plt.show()

X_train_transform = genetic_extractor.transform(X_train_anomaly)
distances_test = genetic_extractor.transform(X_test_anomaly)

pIndex = y_train_anomaly == 1
nIndex = y_train_anomaly == -1

plt.figure(1)
plt.title("GunPoint with first 2 shapelets (train), K=3, L=30")

plt.scatter(X_train_transform[pIndex], np.ones(len(X_train_transform[pIndex])),  facecolor='C0', marker='o', s=100, linewidths=2,
                    edgecolor='black', zorder=2)
plt.scatter(X_train_transform[nIndex], np.ones(len(X_train_transform[nIndex])), facecolor='C3', marker='o', s=100, linewidths=2,
                    edgecolor='black', zorder=2)
plt.legend(["Normal", "Anomalies"], ncol=1, loc='upper left', edgecolor='black',
                      markerscale=1.2, fancybox=True)

plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")

plt.show()


