"""This script implements the Linear Regression Algorithm from scratch."""

# importing required libraries
import numpy as np


# Constructing the linear regression model

class LinearRegression:
    # pylint: disable=too-many-instance-attributes
    """Linear Regression Class"""

    def __init__(self, learning_rate=0.05, iterations=1000):
        """
        Linear Regression Constructor

        Args:
            learning_rate (float, optional): Defaults to 0.05.
            iterations (int, optional): Defaults to 1000.

        Functions:
            fit : (data, label) -> Train the data
            predict : (data) -> Predict the new label from the new data

        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, data, label):
        """
        This function trains a linear regression model.

        Parameters
        ----------
            data : array-like, shape = [n_samples, n_features]
                Training vectors, with n_samples  and n_features.
            label : array-like, shape = [n_samples, n_target_values]
                Target values, with n_samples and n_target_values.

        Returns
        -------
            self : object
        """
        n_samples, n_features = data.shape

        # init parameters
        self.weights, self.bias = np.zeros(n_features), 0

        # Gradient Descent
        for _ in range(self.iterations):
            # Linear forward propagation
            label_pred = np.dot(data, self.weights) + self.bias

            # Compute gradients
            grad_weights = (1 / n_samples) * \
                np.dot(data.T, (label_pred - label))
            grad_bias = (1 / n_samples) * np.sum(label_pred - label)

            # Update parameters
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

    def predict(self, data):
        """ Predict the new data from the previously trained model

        Parameters
        ----------
            data : array-like, shape = [n_samples, n_features]
                Testing vectors, with n_samples and n_features.

        Returns
        -------
            label_pred : array-like, shape = [n_samples, n_target_values]
                Predicted values using the trained linear model.
        """
        return np.dot(data, self.weights) + self.bias

    def r2_score(self, data, label):
        """ Compute the R2 score of the model

        Parameters
        ----------
            data : array-like, shape = [n_samples, n_features]
                Training/Testing vectors, with n_samples and n_features.
            label : array-like, shape = [n_samples, n_target_values]
                Training/Testing target values, with n_samples and n_features.

        Returns
        -------
            r2_score : float
                R2 score of the trained model on the provided data.
        """
        label_pred = self.predict(data)
        corr_matridata = np.corrcoef(label, label_pred)
        corr = corr_matridata[0, 1]
        return corr**2
