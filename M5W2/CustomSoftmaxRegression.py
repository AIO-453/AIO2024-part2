import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class CustomSoftmaxRegression:
    def __init__(self, X_train, X_val, y_train, y_val, learning_rate=0.01, epochs=30, batch_size=1024):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_features = X_train.shape[1]
        self.n_classes = y_train.shape[1]

        self.theta = np.random.uniform(size=(self.n_features, self.n_classes))

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1)[:, None]

    def predict(self, X):
        z = np.dot(X, self.theta)
        return self.softmax(z)

    def compute_loss(self, y_hat, y):
        n = y.size
        return (-1/n) * np.sum(y * np.log(y_hat))

    def compute_gradient(self, X, y, y_hat):
        n = y.size
        return (1/n) * np.dot(X.T, y_hat - y)

    def update_theta(self, gradient):
        return self.theta - self.learning_rate * gradient

    def compute_accuracy(self, X, y):
        y_hat = self.predict(X)
        # acc = (np. argmax (y_hat , axis =1) == np. argmax (y, axis =1) ) . mean()
        acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean()
        return acc

    def fit(self):
        for epoch in range(self.epochs):
            train_batch_losses = []
            train_batch_accs = []
            val_batch_losses = []
            val_batch_accs = []

            for i in range(0, self.X_train.shape[0], self.batch_size):
                # print(i)
                X_batch = self.X_train[i:i+self.batch_size]
                y_batch = self.y_train[i:i+self.batch_size]

                y_hat = self.softmax(np.dot(X_batch, self.theta))

                loss = self.compute_loss(y_hat, y_batch)

                acc = self.compute_accuracy(X_batch, y_batch)

                gradient = self.compute_gradient(X_batch, y_batch, y_hat)

                self.theta = self.update_theta(gradient)

                train_batch_losses.append(loss)
                train_batch_accs.append(acc)

                y_val_hat = self.softmax(np.dot(self.X_val, self.theta))

                val_loss = self.compute_loss(y_val_hat, self.y_val)
                val_batch_losses.append(val_loss)

                val_acc = self.compute_accuracy(self.X_val, self.y_val)
                val_batch_accs.append(val_acc)

            train_batch_loss = np.mean(train_batch_losses)
            train_batch_acc = np.mean(train_batch_accs)
            val_batch_loss = np.mean(val_batch_losses)
            val_batch_acc = np.mean(val_batch_accs)

            self.train_losses.append(train_batch_loss)
            self.val_losses.append(val_batch_loss)
            self.train_accs.append(train_batch_acc)
            self.val_accs.append(val_batch_acc)

            print(f'\nEPOCH {epoch + 1}:\tTraining loss : {
                  train_batch_loss:.3f}\tValidation loss : {val_batch_loss:.3f}')

    def plot(self):
        fig, ax = plt . subplots(2, 2, figsize=(12, 10))
        ax[0, 0].plot(self.train_losses)
        ax[0, 0].set(xlabel='Epoch ', ylabel='Loss ')
        ax[0, 0].set_title('Training Loss ')

        ax[0, 1].plot(self.val_losses, 'orange')
        ax[0, 1].set(xlabel='Epoch ', ylabel='Loss ')
        ax[0, 1].set_title('Validation Loss ')

        ax[1, 0].plot(self.train_accs)
        ax[1, 0].set(xlabel='Epoch ', ylabel='Accuracy ')
        ax[1, 0].set_title('Training Accuracy ')

        ax[1, 1].plot(self.val_accs, 'orange')
        ax[1, 1].set(xlabel='Epoch ', ylabel='Accuracy ')
        ax[1, 1].set_title('Validation Accuracy ')
        plt.show()


# sofmax_regression = CustomSoftmaxRegression(X_train, X_val, y_train, y_val)
# sofmax_regression.fit()

# sofmax_regression = CustomSoftmaxRegression(X_train, X_val, y_train, y_val, lr,epochs,batch_size)
# sofmax_regression.fit()
