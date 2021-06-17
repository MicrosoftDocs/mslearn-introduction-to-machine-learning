'''
This creates figures for content. It is not intended to act as learning material
'''
from numpy.core.function_base import linspace
import graphing
import pandas
import plotly.io
import numpy as np
import statsmodels.formula.api as smf



# FIGURE 1
# Example of a logistic regression module vs a straight line
def logistic_reg(x):
    return 1 / (1+np.exp(-(x-5)*1.2))

fig = graphing.line_2D([("Logistic", logistic_reg), ("Linear", lambda x: x/8 )], x_range=[0,10]) #, trendline=logistic_reg)
plotly.io.write_image(fig, f"2a - unit 02 - figure 1.svg")
plotly.io.write_image(fig, f"2a - unit 02 - figure 1.svg")

# FIGURE 2
# Example of a thresholding a logistic regression module
fig = graphing.line_2D([("Logistic Function", logistic_reg)], label_x="feature", label_y="probability", x_range=[0,10])

# Add boxes
color = graphing.colours_trendline[2][4:-1].split(',')
color = f"rgba({color[0]}, {color[1]}, {color[2]}, 0.3)"
fig.add_shape(type="rect",
    x0=0, y0=0, x1=5, y1=0.5,
    line=dict(
        width=0,
    ),
    fillcolor=color,
)
fig.add_shape(type="rect",
    x0=5, y0=0.5, x1=10, y1=1,
    line=dict(
        width=0,
    ),
    fillcolor=color,
)

plotly.io.write_image(fig, f"2a - unit 02 - figure 2.svg")

# ----- Multiple features, if needed
# Import the data from the .csv file
dataset = pandas.read_csv('Data/avalanche.csv', delimiter="\t")
print(dataset.head())

# Perform logistic regression.
model = smf.logit("avalanche ~ weak_layers + surface_hoar", dataset).fit()
print(model.summary())

def pred(x, y):
    return model.predict(pandas.DataFrame(dict(weak_layers=[x], surface_hoar=[y])))

graphing.surface(np.array(linspace(0,20,40)),np.array(np.array(linspace(0,20,40))),pred, show=True)

