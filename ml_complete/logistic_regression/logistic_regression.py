"""This script implements the Logistic Regression Algorithm from scratch."""

# importing required libraries
import numpy as np


# Constructing the Logistic regression model

class LogisticRegression:
    """Logistic Regression Class"""
    
    def __init__(self, learning_rate=0.05, iterations=1000):
        """
        Logistic Regression

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
        This function trains a logistic regression model.

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
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        pred_class = np.where(y_pred >= 0.5, 1, 0)
        return np.array(pred_class, dtype=int)

    def sigmoid(self, z):
        """ Compute the sigmoid function of the input z
        
        Parameters
        ----------
            z : array-like, shape = [n_samples, n_features]
                Input values.
        
        Returns
        -------
            sigmoid : array-like, shape = [n_samples, n_features]
                Sigmoid values of the input z.
        """
        return 1 / (1 + np.exp(-z))

    def accuracy_score(self, X, y):
        """ Compute the Accuracy Score of the model

        Parameters
        ----------
            X : array-like, shape = [n_samples, n_features]
                Training/Testing vectors, with n_samples and n_features.
            y : array-like, shape = [n_samples, n_target_values]
                Training/Testing target values, with n_samples and n_features.

        Returns
        -------
            Accuracy_score : float
                Accuracy Score of the trained model on the provided X.
        """
        y_pred = self.predict(X)
        y_pred = np.round(y_pred)
        return np.mean(y == y_pred)
