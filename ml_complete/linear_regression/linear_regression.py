"""This script implements the Linear Regression Algorithm from scratch."""

# importing required libraries
import numpy as np
import matplotlib.pyplot as plt


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


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    n_sample = int(input("Enter the number of samples: "))
    n_features = int(input("Enter the number of features: "))

    # Prepare the data
    data, label = datasets.make_regression(
        n_samples=n_sample, n_features=n_features, noise=10, random_state=4
    )
    print(data.shape, label.shape)

    # Split the data into training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, test_size=0.2, random_state=0
    )

    # Train the model
    model = LinearRegression(0.05, 1000)
    model.fit(data_train, label_train)

    # Predict the values
    label_pred = model.predict(data_test)

    # Compute the R2 score
    r2_score = model.r2_score(data_test, label_test)
    print("R2 score: ", r2_score)

    # Plot the graph for data with 1 feature
    if data.shape[1] == 1:
        # Plot the data
        cmap = plt.get_cmap("viridis")
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(
            data_train, label_train, color=cmap(0.9), s=10, label="Train Data")
        plt.scatter(
            data_test, label_test, color=cmap(0.5), s=10, label="Test Data")
        plt.plot(
            data_test, label_pred, color="red", label="Prediction")
        plt.datalabel("data - Features", fontsize=18)
        plt.ylabel("label - Outcome", fontsize=18)
        plt.legend()
        plt.show()
    else:
        print("Plotting not possible for more than 1 feature in a 2-D plot")
