"""This script implements the Linear Regression Algorithm from scratch."""

# importing required libraries
import numpy as np


# Constructing the linear regression model

class LinearRegression:
    """Linear Regression Class"""

    def __init__(self, learning_rate=0.05, iterations=1000):
        """
        Linear Regression Constructor

        Args:
            learning_rate (float, optional): Defaults to 0.05.
            iterations (int, optional): Defaults to 1000.

        Functions:
            fit : (X, y) -> Train the X
            predict : (X) -> Predict the new y from the new X

        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        This function trains a linear regression model.

        Parameters
        ----------
            X : array-like, shape = [n_samples, n_features]
                Training vectors, with n_samples  and n_features.
            y : array-like, shape = [n_samples, n_target_values]
                Target values, with n_samples and n_target_values.

        Returns
        -------
            self : object
        """
        n_samples, n_features = X.shape

        # init parameters
        self.weights, self.bias = np.zeros(n_features), 0

        # Gradient Descent
        for _ in range(self.iterations):
            # Linear forward propagation
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            grad_weights = (1 / n_samples) * \
                np.dot(X.T, (y_pred - y))
            grad_bias = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

    def predict(self, X):
        """ Predict the new X from the previously trained model

        Parameters
        ----------
            X : array-like, shape = [n_samples, n_features]
                Testing vectors, with n_samples and n_features.

        Returns
        -------
            y_pred : array-like, shape = [n_samples, n_target_values]
                Predicted values using the trained linear model.
        """
        return np.dot(X, self.weights) + self.bias

    def r2_score(self, X, y):
        """ Compute the R2 score of the model

        Parameters
        ----------
            X : array-like, shape = [n_samples, n_features]
                Training/Testing vectors, with n_samples and n_features.
            y : array-like, shape = [n_samples, n_target_values]
                Training/Testing target values, with n_samples and n_features.

        Returns
        -------
            r2_score : float
                R2 score of the trained model on the provided X.
        """
        y_pred = self.predict(X)
        corr_matriX = np.corrcoef(y, y_pred)
        corr = corr_matriX[0, 1]
        return corr**2
