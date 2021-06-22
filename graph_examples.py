'''
These are examples on how to use graphing.py
Pay attention to save_charts and show_charts
If show_charts is True, graphs will open in your browser
If save_charts is True, graphs will save to svg
'''
import pandas
import numpy as np
import graphing
import os
import plotly.io

save_charts = True
show_charts = False
dir_save = "charts/"
chart_count = 0

if save_charts and not os.path.exists(dir_save):
    os.mkdir(dir_save)
    

def save_chart(fig):
    if save_charts:
        global chart_count
        chart_count += 1
        plotly.io.write_image(fig, dir_save + str(chart_count) + ".svg")


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

# Make example graphs.

def line_1_eq(x):
    return 0.02 * x ** 2 + 0.1 * x + 1

def line_2_eq(x):
    return -0.02 * (x - 2) ** 2 + 0.1 * x + 2

def line_3_eq(x):
    return -0.04 * (x + 2) ** 2 + 0.2 * x - 2

# Plot single function
fig = graphing.line_2D(("line number one", line_1_eq), 
                x_range=[-10,10], 
                label_x="x-axis", 
                label_y="Goodness (cm)",
                show=show_charts)
save_chart(fig)


# Plot multiple functions with specific x-values
fig = graphing.line_2D([  ("line number one", line_1_eq), 
                    ("line number two", line_2_eq), 
                    ("line number three", line_3_eq)], 
                x_range=[-10, 2, 5, 10], 
                label_x="x-axis", 
                label_y="Goodness (cm)",
                legend_title="Line number",
                title="Line plot", show=show_charts)
save_chart(fig)


# Plot multiple functions with specific x-values and precalculated values
dat = dict(line_number_one=[1,2,3,4,5], line_number_two=[2, 3,5,7,11])
fig = graphing.line_2D(dat, 
                x_range=[0, 1, 2, 3, 4], 
                label_x="x-axis", 
                legend_title="Line number",
                title="Line plot", show=show_charts)
save_chart(fig)

def surf_eq(x, y):
    return x * 8 + y ** 2
fig = graphing.surface(np.array([0,1]), np.linspace(-2,2), surf_eq, axis_title_x="X values", axis_title_y="Y Values", axis_title_z="Z values", show=show_charts)
save_chart(fig)


for n in [10,50,200,1000]:
    fig = graphing.scatter_2D(dataset[:n], title=f"A 2D scatter with {n} points", show=show_charts, trendline=shoe_to_height)
    save_chart(fig)

fig = graphing.scatter_3D(dataset, title="A 3D scatter", show=show_charts)
save_chart(fig)

fig = graphing.box_and_whisker(dataset, title="A simple box and whisker", show=show_charts)
save_chart(fig)
fig = graphing.box_and_whisker(dataset, label_x='hat_size', title="A box and whisker, 1 group level", show=show_charts)
save_chart(fig)
fig = graphing.box_and_whisker(dataset, label_x='hat_size', label_x2='hair_colour', title="A box and whisker, 2 group levels", show=show_charts)
save_chart(fig)

fig = graphing.histogram(dataset, title="A histogram (one variable)", show=show_charts)
save_chart(fig)

fig = graphing.histogram(dataset, label_x="hat_size", label_y="shoe_size", title="A histogram (two variables)", show=show_charts)
save_chart(fig)

fig = graphing.histogram(dataset, label_colour="hat_size", title="A stacked histogram", show=show_charts)
save_chart(fig)

fig = graphing.histogram(dataset, title="A histogram (one variable + boxplot)", include_boxplot=True, show=show_charts)
save_chart(fig)

fig = graphing.histogram(dataset, label_colour="hat_size", title="A stacked histogram (+ boxplot)", include_boxplot=True, show=show_charts)
save_chart(fig)

fig = graphing.multiple_histogram(dataset, label_x="hat_size", label_y="is_male", histfunc='avg', label_group="hair_colour", title="A histogram (two variables)", show=show_charts)
save_chart(fig)

fig = graphing.multiple_histogram(dataset, label_x="hat_size", label_group="hair_colour", title="A histogram (two variables)", show=show_charts)
save_chart(fig)


# saves plot as a static file 
if save_chart:
    # Use defaults
    graphing.save_plot_as_image(fig, dir_save + "test_plot.jpg")

    # Set custom size
    graphing.save_plot_as_image(fig, dir_save + "./test_plot2.jpg", width=350, height=200)

    # Set custom scale
    graphing.save_plot_as_image(fig, dir_save + "./test_plot3.jpg", scale=2)

    # Save as PNG
    graphing.save_plot_as_image(fig, dir_save + "./test_plot4.png", format="png")

    # Save as PDF
    graphing.save_plot_as_image(fig, dir_save + "./test_plot5.pdf", format="pdf")
