import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from GENDIS_python.gendis.genetic import GeneticExtractor
from trials.preprocessing_anomaly import preprocessing_anomaly

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("ECGFiveDays")

X_train, y_train, X_test, y_test = preprocessing_anomaly(X_train, y_train, X_test, y_test, alpha=0.05, normal_class=1)

X_train = TimeSeriesScalerMinMax().fit_transform(X_train)[:,:,0]
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)[:,:,0]

genetic_extractor = GeneticExtractor(population_size=50, iterations=25, verbose=True,
                                     mutation_prob=0.3, crossover_prob=0.3,
                                     wait=10, max_len=len(X_train) // 2)
### NOTE: when extracting shapelets, 'avg' is the average fitness value, which INCREASES as more populations are generated

shapelets = genetic_extractor.fit(X_train, y_train)
distances_train = genetic_extractor.transform(X_train)
distances_test = genetic_extractor.transform(X_test)

lr = LogisticRegression()
lr.fit(distances_train, y_train)


print('Balanced Accuracy = {}'.format(balanced_accuracy_score(y_test, lr.predict(distances_test))))

import matplotlib.pyplot as plt
plt.figure()
for shap in genetic_extractor.shapelets:
    plt.plot(shap)
plt.title('Extracted shapelets - ECGFiveDays')
plt.show()