import pandas
import numpy as np
import graphing

# Make some fake data
def shoe_to_height(x):
    return x*x

shoe_size = np.random.standard_normal(size=10000)
height = shoe_to_height(shoe_size) + (np.random.rand(shoe_size.shape[0]) - 0.5) * 0.2
hat_size = np.random.randint(0,5, size=shoe_size.shape[0])
hair_colour = np.tile(['blonde', 'brunette'], int(shoe_size.shape[0]/2))
is_male = np.random.randint(0, 2, size=shoe_size.shape[0])
data = {
    'shoe_size' : shoe_size,
    'person_height': height,
    'hat_size': hat_size,
    'hair_colour': hair_colour,
    'is_male': is_male
}

# Convert it into a table using pandas
dataset = pandas.DataFrame(data)

# Make example graphs. These will open in your browser

def line_1_eq(x):
    return 0.02 * x ** 2 + 0.1 * x + 1

def line_2_eq(x):
    return -0.02 * (x - 2) ** 2 + 0.1 * x + 2

def line_3_eq(x):
    return -0.04 * (x + 2) ** 2 + 0.2 * x - 2

# Plot single function
graphing.line_2D(("line number one", line_1_eq), 
                x_range=[-10,10], 
                label_x="x-axis", 
                label_y="Goodness (cm)",
                show=True)


# Plot multiple functions
graphing.line_2D([  ("line number one", line_1_eq), 
                    ("line number two", line_2_eq), 
                    ("line number three", line_3_eq)], 
                x_range=[-10,10], 
                label_x="x-axis", 
                label_y="Goodness (cm)",
                legend_title="Line number",
                title="Line plot", show=True)


def surf_eq(x, y):
    return x * 8 + y ** 2
graphing.surface(np.array([0,1]), np.linspace(-2,2), surf_eq, axis_title_x="X values", axis_title_y="Y Values", axis_title_z="Z values", show=True)

for n in [10,50,200,1000]:
    graphing.scatter_2D(dataset[:n], title=f"A 2D scatter with {n} points", show=True, trendline=shoe_to_height)

graphing.scatter_3D(dataset, title="A 3D scatter", show=True)

graphing.box_and_whisker(dataset, title="A simple box and whisker", show=True)
graphing.box_and_whisker(dataset, label_x='hat_size', title="A box and whisker, 1 group level", show=True)
graphing.box_and_whisker(dataset, label_x='hat_size', label_x2='hair_colour', title="A box and whisker, 2 group levels", show=True)

graphing.histogram(dataset, title="A histogram (one variable)", show=True)
graphing.histogram(dataset, label_x="hat_size", label_y="shoe_size", title="A histogram (two variables)", show=True)
graphing.histogram(dataset, label_colour="hat_size", title="A stacked histogram", show=True)

graphing.histogram(dataset, title="A histogram (one variable + boxplot)", include_boxplot=True, show=True)
graphing.histogram(dataset, label_colour="hat_size", title="A stacked histogram (+ boxplot)", include_boxplot=True, show=True)

graphing.multiple_histogram(dataset, label_x="hat_size", label_y="is_male", histfunc='avg', label_group="hair_colour", title="A histogram (two variables)", show=True)
graphing.multiple_histogram(dataset, label_x="hat_size", label_group="hair_colour", title="A histogram (two variables)", show=True)