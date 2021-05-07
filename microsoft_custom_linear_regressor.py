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

    def fit(self, X, y, calculate_error):
        '''
        Fits the model

        X: data to feed the mode
        y: expected predictions
        calculate_error: A function that calculates
        '''

        # Create a cost function. We use this in a minute.
        # Read the comments to understand what this does 
        def cost_function(coefficients):
            '''
            This makes a prediction about y, using coefficients 
            (slope and intercept). It then uses the calculate_error 
            function (provided) to calculate the error term.
            '''
            # Get our line's slope and intercept
            # from the coefficients
            slope = coefficients[0]
            intercept = coefficients[1]

            # Predict Y from X
            y_predicted = X * slope + intercept

            # Calculate the error versus the expected value
            return calculate_error(y, y_predicted)

        # we want to fit two coefficients. One for slope, and 
        # one for intercept. First, make an initial guess 
        # for these values. Let's guess 0 and 0
        coefs_to_fit = np.array([0,0])

        # Call scipy's minimize function. This will try to
        # find the best parameters that minimise the cost
        # function.  (calculate_error)
        final_coefficients = minimize(cost_function, coefs_to_fit)

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
