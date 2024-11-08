import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import CustomSoftmaxRegression as sm

if __name__ == "__main__":
    dataset_path = './creditcard.csv'
    df = pd.read_csv(dataset_path)
    # print(df.head())

    dataset_arr = df.to_numpy()
    X, y = dataset_arr[:, :-
                       1].astype(np.float64), dataset_arr[:, -1].astype(np.uint8)

    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    n_classes = np.unique(y, axis=0).shape[0]
    n_samples = y.shape[0]

    # y_encoded = np.array([
    #     [np. zeros ( n_classes ) for _ in range ( n_samples )]
    # ])
    y_encoded = np.zeros((n_samples, n_classes))
    y_encoded[np.arange(n_samples), y] = 1

    val_size = 0.2
    test_size = 0.125
    random_state = 2
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(X_b, y_encoded,
                                                      test_size=val_size,
                                                      random_state=random_state,
                                                      shuffle=is_shuffle)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    shuffle=is_shuffle)

    normalizer = StandardScaler()
    X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
    X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
    X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

    lr = 0.01
    epochs = 30
    batch_size = 1024
    n_features = X_train.shape[1]

    np.random.seed(random_state)
    # theta = np.random.uniform(size =( n_features , n_classes ))
    # theta

    sofmax_regression = sm.CustomSoftmaxRegression(
        X_train, X_val, y_train, y_val, lr, epochs, batch_size)
    sofmax_regression.fit()

    val_set_acc = sofmax_regression.compute_accuracy(X_val, y_val)
    test_set_acc = sofmax_regression.compute_accuracy(X_test, y_test)
    print('Evaluation on validation and test set :')
    print(f'Accuracy : {val_set_acc}')
    print(f'Accuracy : {test_set_acc}')

    sofmax_regression.plot()
