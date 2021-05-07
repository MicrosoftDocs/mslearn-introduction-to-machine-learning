import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class SimpleLinearRegression:
    """
    

    weights, J_history = model.gradientDescent(X, y, learning_rate, num_iters)
    """

    # np.seterr(divide='ignore', invalid='ignore')


    def cost(self, X, y, weights):
        """Cost Function"""
        m = y.shape[0]
        J = 0

        difference = np.dot(X, weights) - y
        J = np.dot(difference.T, difference) / (2 * m)

        return J

    def fit(self, X, y, learning_rate, num_iters):
        """Performs Gradient Descent"""
        # X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        # Initialize weights
        weights = np.zeros(X.shape[1])
        m = y.shape[0]

        # make a copy of weights, which will be updated by gradient descent
        weights = weights.copy()

        J_history = []

        for i in range(num_iters):
            h_vec = np.dot(X, weights)
            # print(f"lr {learning_rate}")
            # print(f"m {m}")
            # print(f"h_vec {h_vec}")
            # np.nan_to_num(h_vec, copy=True, nan=0.00001, posinf=None, neginf=None)
            # print(h_vec)
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
        # X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        weights = self.weights
        return np.dot(X, weights)
