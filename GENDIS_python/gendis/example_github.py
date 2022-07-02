import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

from genetic import GeneticExtractor


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(1337)  # Random seed for reproducibility

from tslearn.datasets import UCR_UEA_datasets

# Load ItalyPowerDemand dataset
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ItalyPowerDemand')
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

# Visualize the timeseries in the train and test set
colors = ['r', 'b', 'g', 'y', 'c']

plt.figure(1, figsize=(10, 5))
for ts, label in zip(X_train, y_train):
    plt.plot(range(len(ts)), ts, c=colors[label%len(colors)])
plt.title('The timeseries in the train set')
plt.show()

plt.figure(2, figsize=(10, 5))
for ts, label in zip(X_test, y_test):
    plt.plot(range(len(ts)), ts, c=colors[label%len(colors)])
plt.title('The timeseries in the test set')
plt.show()

# Extract the shapelets using the train set. Enabled verbosity.
# Plotting is possible too, by setting plot='notebook' in jupyter or True else
genetic_extractor = GeneticExtractor(verbose=True, population_size=50, iterations=10, plot=None) # location=True
genetic_extractor.fit(X_train, y_train)

plt.figure(3)
for shap in genetic_extractor.shapelets:
    plt.plot(shap)
plt.title('The extracted shapelets')
plt.show()

x, y, y_err, y_max = [], [], [], []
for it, stat in genetic_extractor.history:
    x.append(it)
    y.append(stat['avg'])
    y_err.append(stat['std'] ** 1.25)
    y_max.append(stat['max'])
x, y, y_err, y_max = np.array(x), np.array(y), np.array(y_err), np.array(y_max)

plt.figure(4)
plt.plot(x, y, 'k-', label='Average')
plt.fill_between(x, y-y_err, y+y_err, label='Variance')
plt.plot(x, y_max, 'k--', label='Maximum')
plt.title('The fitness (negative logloss) in function of the number of generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()

# Calculate distances from timeseries to extracted shapelets and fit a Logistic Regression model
distances_train = genetic_extractor.transform(X_train)
distances_test = genetic_extractor.transform(X_test)

lr = LogisticRegression(random_state=2020)
lr.fit(distances_train, y_train)

# Print the accuracy score on the test set
print('Accuracy LR = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))