import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class PolynomialLinearRegression:
    """
    Custom Polynomial Linear Regression Implementation

    This allows us to set the learning rate and number of iterations in training
    and encapsulates feature normalization and feature matrix convertion to polynomial


    To return weights and cost history, use

    weights, J_history = model.gradientDescent(X, y, learning_rate, num_iters)
    """

    # np.seterr(divide='ignore', invalid='ignore')

    def to_polynomial(self, X):
        """Convert feature matrix into a 2 degree polynomial matrix"""
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        return X_poly

    def normalize(self, X):
        """Normalize features using mean and std"""
        # You need to set these values correctly
        X_norm = X.copy()
        mu = np.zeros(X.shape[1])
        sigma = np.zeros(X.shape[1])
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X_norm = (X_norm - mu) / sigma

        # not needed as this is done by to_polynomial
        # X_norm = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

        return X_norm, mu, sigma

    def cost(self, X, y, weights):
        """Cost Function"""
        m = y.shape[0]
        J = 0

        difference = np.dot(X, weights) - y
        J = np.dot(difference.T, difference) / (2 * m)

        return J

    def fit(self, X, y, learning_rate, num_iters):
        """Normalizes X and performs Gradient Descent"""

        # Normalize features
        X_norm, mu, sigma = self.normalize(X)

        # Convert features to polynomial
        X = self.to_polynomial(X_norm)

        # Initialize weights
        weights = np.zeros(X.shape[1])
        m = y.shape[0]

        # make a copy of weights, which will be updated by gradient descent
        weights = weights.copy()

        J_history = []

        for i in range(num_iters):
            h_vec = np.dot(X, weights)
            weights = weights - (learning_rate / m) * np.dot(h_vec - y, X)

            # Append current cost to history
            J_history.append(self.cost(X, y, weights))

        self.weights = weights

        return weights, J_history

    def predict(self, X):
        """ 
        Generates predictions for a 'X' after we have trained the model and gathered weights 
        X is normalized and polynomial features are added here too.
        """
        weights = self.weights
        # Normalize features
        X_norm, mu, sigma = self.normalize(X)

        # Convert features to polynomial
        X = self.to_polynomial(X_norm)

        return np.dot(X, weights)
