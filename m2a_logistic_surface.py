'''
The following code is not intended as learning material.
It creates 3D surface plots for a logistic regression model
'''
import pandas
import numpy as np
import graphing

def logistic_to_surface_plot(model, plot_features, data):
    '''Plots two features of a logistic regression model as a surface'''

    # Give status as this can take several seconds to run
    print("Creating plot...")

    
    other_features = [f for f in data.columns if f not in plot_features]

    means = np.average(data[other_features], axis=0)
    mins = np.min(data[plot_features], axis=0)
    maxes = np.max(data[plot_features], axis=0)

    df = pandas.DataFrame()

    for f,m in zip(other_features, means):
        df[f] = [m]

    def predict(x, y):
        '''
        Makes a prediction using the model
        '''
        df[plot_features[0]] = [x]
        df[plot_features[1]] = [y]

        return model.predict(df)

    # Create a 3d plot of predictions
    x_vals = np.array(np.linspace(mins[plot_features[0]], maxes[plot_features[0]],20))
    y_vals = np.array(np.linspace(mins[plot_features[1]], maxes[plot_features[1]],20))

    return graphing.surface(x_vals, 
                    y_vals, 
                    predict, 
                    title="Model Prediction", 
                    axis_title_x=plot_features[0], 
                    axis_title_y=plot_features[1], 
                    axis_title_z="Probability")