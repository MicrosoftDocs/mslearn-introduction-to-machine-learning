'''
This script creates figures for module 2c. It is not intended as learning material
'''

from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import graphing
import numpy.random

# This is just some hypothetical relationship between a 
# model parameter and a cost function. It is not intended
# to represent anything tangibly real
my_eq = lambda x: (10 - (4*(x)**0.5) - 0.1 *(6-x))/10

def logloss(x):
    '''
    Log loss as response to model parameter A (purely hypothetical)
    '''
    return my_eq(x)

def acc(x):
    '''
    Accuracy as response to model parameter A (purely hypothetical)
    '''
    return 1 - np.round(my_eq(x)*10)/10

fig = graphing.line_2D([("Log loss",logloss), ("Accuracy", acc)], np.linspace(0,5,10000), label_x="Value of model parameter A", label_y="cost")
graphing.save_plot_as_image(fig, "2c - unit 06 - figure 1.jpg", width=400, height=None)
