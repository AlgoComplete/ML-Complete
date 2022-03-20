# Author: Indrashis Paul
# Date: 15-03-2022

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score


# Constructing the Logistic regression model

class LogisticRegression:
    def __init__(self, learning_rate=0.05, iterations=1000):
        """
        Logistic Regression

        Args:
            learning_rate (float, optional): Defaults to 0.05.
            iterations (int, optional): Defaults to 1000.

        Functions:
            fit : (X, y) -> Train the data
            predict : (X) -> Predict the new data from the previous trained model

        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        This function takes the X and y as input and trains the model based on the defined classmethods.

        Parameters
        ----------
            X : array-like, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and n_features is the number of features.
            y : array-like, shape = [n_samples, n_target_values]
                Target values, where n_samples is the number of samples and n_target_values is the number of target values.

        Returns
        -------
            self : object
        """
        n_samples, n_features = X.shape

        # init parameters
        self.weights, self.bias = np.zeros(n_features), 0

        # Gradient Descent
        for i in range(self.iterations):
            # Linear forward propagation
            linear_model = np.dot(X, self.weights) + self.bias

            # Sigmoid function
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            grad_weights = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            grad_bias = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        pred_class = np.where(y_pred >= 0.5, 1, 0)
        return np.array(pred_class, dtype=int)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def accuracy_score(self, X, y):
        y_pred = self.predict(X)
        y_pred = np.round(y_pred)
        return np.mean(y == y_pred)


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    n_sample = int(input("Enter the number of samples: "))
    n_features = int(input("Enter the number of features: "))

    n_redundant = 0
    if n_features > 1:
        n_redundant = 1

    # Prepare the data
    X, y = datasets.make_classification(
        n_samples=n_sample, n_features=n_features, n_classes=2, n_informative=1, n_redundant=n_redundant, n_repeated=0, n_clusters_per_class=1, random_state=4
    )

    print(X.shape, y.shape)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Train the model
    model = LogisticRegression(0.05, 1000)
    model.fit(X_train, y_train)

    # Predict the values
    y_pred = model.predict(X_test)

    # Compute the R2 score
    accuracy_score = model.accuracy_score(X_test, y_test)
    print("Accuracy score: ", accuracy_score)

    def plot_data(x, y):
        plt.xlabel('score of test-1')
        plt.ylabel('score of test-1')
        for i in range(x.shape[0]):
            if y[i] == 1:
                plt.plot(x[i, 0], x[i, 1], 'gX')
            else:
                plt.plot(x[i, 0], x[i, 1], 'mD')
        plt.show()

    plot_data(X_test, y_test)

    # Plot the graph for data with 1 feature
    if (X.shape[1] == 1):
        # Plot the data
        cmap = plt.get_cmap("viridis")
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(
            X_train, y_train, color=cmap(0.9), s=10, label="Training Data")
        plt.scatter(
            X_test, y_test, color=cmap(0.5), s=10, label="Testing Data")
        plt.plot(
            X_test, y_pred, color="red", label="Prediction")
        plt.xlabel("X - Features", fontsize=18)
        plt.ylabel("y - Outcome", fontsize=18)
        plt.legend()
        plt.show()
    else:
        print("Plotting the data is not possible for more than 1 feature in a 2-D plot")