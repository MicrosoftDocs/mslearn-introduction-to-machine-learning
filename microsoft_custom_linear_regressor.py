from sklearn.base import BaseEstimator
from scipy.optimize import minimize
import numpy as np


class MicrosoftCustomLinearRegressor(BaseEstimator):
    '''
    This class runs linear regression using any custom
    error function. 

    Example use:
    ```
    X = ...
    y = ...
    my_error_function = lambda y,y_predicted: np.sum((y-y_predicted)**2)
    my_model = MicrosoftCustomLinearRegressor().fit(X, y, my_error_function)
    ```
    '''
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y, cost_function):
        '''
        Fits the model

        X: data to feed the mode
        y: expected predictions
        cost_function: A function that calculates error
        '''

        def estimate_and_calc_error(coefficients):
            '''
            This makes a prediction about y, using coefficients 
            (slope and intercept). It then uses the cost_function 
            function (provided) to calculate the error term.
            '''
            # Get our line's slope and intercept
            # from the coefficients
            slope = coefficients[0]
            intercept = coefficients[1]

            # Predict Y from X
            y_predicted = X * slope + intercept

            # Calculate the error versus the expected value
            return cost_function(y, y_predicted)

        # we want to fit two coefficients. One for slope, and 
        # one for intercept. First, make an initial guess 
        # for these values. Let's guess 0 and 0
        coefs_to_fit = np.array([0,0])

        # Call scipy's minimize function. This will try to
        # find the best parameters that minimise the cost
        # function.  (cost_function)
        final_coefficients = minimize(estimate_and_calc_error, coefs_to_fit)

        # Save the coefficients so we can use them
        # in the predict method
        self.slope = final_coefficients.x[0]
        self.intercept = final_coefficients.x[1]

        return self
    
    def predict(self, X):
        '''
        Predicts y from X using the fitted slope and intercept
        '''
        return X * self.slope + self.intercept
