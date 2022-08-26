import numpy as np

def preprocessing_anomaly(X_train, y_train, X_test=None, y_test=None, alpha=0.1, normal_class=1, normal_prop=0.8, valid_prop = 0):
    '''
    Transform the dataset in a new one for anomaly detection according to Beggel's paper
    @param alpha: parameter to control the proportion of normal - anomalies
    @param normal_class: indicate label of the normal class
    @param normal_prop: indicates the proportion of normal instances in training set w.r.t. the total number
                        default is 0.8 that means 80% of normal series go into training set (as in Beggel's paper)
    '''
    if isinstance(X_test, type(None)):
        X_total = X_train
        y_total = y_train
    else:
        X_total = np.concatenate((X_train, X_test), axis=0)
        y_total = np.concatenate((y_train, y_test), axis=0)
    N, M, _ = X_total.shape

    # select the normal class time series
    N_positive_total = len(X_total[y_total == normal_class])
    # take 80% of normal observation to train as in the paper
    size_positive = round(normal_prop * N_positive_total)
    random_indices = np.random.choice(N_positive_total, size=size_positive, replace=False)

    X_train_positive = X_total[y_total == normal_class][random_indices]
    y_train_positive = np.ones(len(X_train_positive), dtype='int')
    X_test_positive = np.delete(X_total[y_total == normal_class], random_indices, axis=0)
    y_test_positive = np.ones(len(X_test_positive), dtype='int')

    # size of the negative samples in training set
    size_negative = round(alpha * len(X_train_positive))
    # take few random negative examples
    random_indices = np.random.choice(N - N_positive_total, size=size_negative, replace=False)
    X_train_negative = X_total[y_total != normal_class][random_indices]
    # collapse all the anomalous ts in one class
    y_train_negative = np.array([-1] * len(random_indices), dtype='int')
    X_test_negative= np.delete(X_total[y_total != normal_class], random_indices, axis=0)
    y_test_negative = np.array([-1] * len(X_test_negative), dtype='int')

    X_train_anomaly = np.concatenate((X_train_positive, X_train_negative), axis=0)
    y_train_anomaly = np.concatenate((y_train_positive, y_train_negative), axis=0)

    X_test_anomaly = np.concatenate((X_test_positive, X_test_negative), axis=0)
    y_test_anomaly = np.concatenate((y_test_positive, y_test_negative), axis=0)

    if valid_prop ==0:
        return X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly
    
    # create validation set
    n_test, _, _ = X_test_anomaly.shape
    size_valid = round(valid_prop * n_test)
    valid_indices = np.random.choice(n_test, size=size_valid, replace=False)

    X_valid_anomaly = X_test_anomaly[valid_indices]
    y_valid_anomaly = y_test_anomaly[valid_indices]

    X_test_anomaly = np.delete(X_test_anomaly, valid_indices, axis=0)
    y_test_anomaly = np.delete(y_test_anomaly, valid_indices, axis=0)

    return X_train_anomaly, y_train_anomaly, X_test_anomaly, y_test_anomaly, X_valid_anomaly, y_valid_anomaly




