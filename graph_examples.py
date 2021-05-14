import pandas
import numpy as np
import graphing

# Make some fake data
def shoe_to_height(x):
    return x*x

shoe_size = hat_size = np.random.standard_normal(size=1000)
height = shoe_to_height(shoe_size) + (np.random.rand(shoe_size.shape[0]) - 0.5) * 0.2
hat_size = np.random.randint(0,5, size=shoe_size.shape[0])
hair_colour = np.tile(['blonde', 'brunette'], int(shoe_size.shape[0]/2))

data = {
    'shoe_size' : shoe_size,
    'person_height': height,
    'hat_size': hat_size,
    'hair_colour': hair_colour
}

# Convert it into a table using pandas
dataset = pandas.DataFrame(data)

# Make example graphs. These will open in your browser
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
