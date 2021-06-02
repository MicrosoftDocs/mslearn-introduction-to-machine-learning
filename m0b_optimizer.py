'''
This describes is an optimiser used in module 0b
This hard codes some things to make the example code simple. In a normal
situation there would be many more options and nothing hard coded
'''
import numpy as np

class MyOptimizer:

    def calculate_gradient(self, model_input, diff):
        """
        This calculates gradient for a linear regession 
        using the SSD cost function:
        cost = sum((predicted - actual)^2)
        """

        # The partial derivatives of SSD are as follows
        # You don't need to be able to do this just yet but
        # it is important to note these give you the two gradients
        # that we need to train our model
        grad_slope = (diff * model_input).sum() * 2
        grad_intercept = diff.sum() * 2
        return grad_intercept, grad_slope


    def get_parameter_updates(self, model_inputs, cost, diff):

        # Calculate the gradient
        grad_intercept, grad_slope = self.calculate_gradient(model_inputs, diff)

        # Update the estimation of the line
        # We have hard coded some learning rates here
        # to keep things simple
        slope_update =     -grad_slope / 1000 / 2100 #5E6
        intercept_update = -grad_intercept / 1000 / 2100#5E6

        return intercept_update, slope_update 