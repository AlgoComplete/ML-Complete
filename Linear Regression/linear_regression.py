# Author: Indrashis Paul
# Date: 15-03-2022

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Constructing the linear regression model

class LinearRegression:
    def __init__(self, learning_rate=0.05, iterations=1000):
        """
        Linear Regression

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
        self.costs = None

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
        self.weights, self.bias, self.costs = np.zeros(n_features), 0, []

        # Gradient Descent
        for i in range(self.iterations):
            # Linear forward propagation
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            grad_weights = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            grad_bias = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def r2_score(self, X, y):
        y_pred = self.predict(X)
        corr_matrix = np.corrcoef(y, y_pred)
        corr = corr_matrix[0, 1]
        return corr**2


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    n_sample = int(input("Enter the number of samples: "))
    n_features = int(input("Enter the number of features: "))

    # Prepare the data
    X, y = datasets.make_regression(
        n_samples=n_sample, n_features=n_features, noise=10, random_state=4
    )
    print(X.shape, y.shape)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Train the model
    model = LinearRegression(0.01, 100)
    model.fit(X_train, y_train)

    # Predict the values
    y_pred = model.predict(X_test)

    # Compute the R2 score
    r2_score = model.r2_score(X_test, y_test)
    print("R2 score: ", r2_score)

    if (X.shape[1] == 1):
        # Plot the data
        cmap = plt.get_cmap("viridis")
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
        plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
        plt.plot(X_test, y_pred, color="red", label="Prediction")
        plt.xlabel("X - Features", fontsize=18)
        plt.ylabel("y - Outcome", fontsize=18)
        plt.show()
    else:
        print("Plotting the data is not possible for more than 1 feature in a 2-D plot")
