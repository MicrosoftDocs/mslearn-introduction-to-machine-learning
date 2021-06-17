'''
This creates figures for content. It is not intended to act as learning material
'''
from numpy.core.function_base import linspace
import graphing
import pandas
import plotly.io
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import log_loss

# FIGURE 1
# Log loss
def loglosses(x):
    return [log_loss([0], [cur], labels=[0,1]) for cur in x]

fig = graphing.line_2D([("Log-Loss", loglosses)], x_range=[0,0.999], label_y="log loss if truth is 0", label_x="prediction", title="")
fig.update_xaxes(range=[0, 1])
plotly.io.write_image(fig, f"2a - unit 04 - figure 1.svg")

# FIGURE 2
def mse(x):
    # Single values and truth of zero, so we just square them
    return x ** 2

fig = graphing.line_2D([("Log-Loss", loglosses), ("MSE", mse)], x_range=[0,0.999], label_y="cost if truth is 0", label_x="prediction", title="")
fig.update_xaxes(range=[0, 1])
plotly.io.write_image(fig, f"2a - unit 04 - figure 2.svg")


exit()

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

