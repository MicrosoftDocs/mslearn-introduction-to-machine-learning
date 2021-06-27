"""
This creates figures for content. It is not intended to act as learning material
"""
# %%

import graphing
import pandas
import numpy
import numpy.random
from scipy.stats import norm, skewnorm


# We need the following **histograms**. When the script is run, it should save
# each as a jpg.


# 1: Randomly distributed data

# Creates 1000 points of normally distributed data
data = norm.rvs(0, 0.1, size=1000)
df = pandas.DataFrame(data, columns=["Label"])

# Plot and save histogram
fig = graphing.histogram(df, title="")
graphing.save_plot_as_image(fig, file="./2c - unit 03 - normal_distribution.jpg", format="jpg")


# %%
# 2: skewed normal distribution - similar to the above distribution but
#  where the positive (right) tail is much longer than the left. The mean should still be 0

# hiher alpha = more skewed to the right
alpha = 4
data = skewnorm.rvs(alpha, size=1000)
df = pandas.DataFrame(data, columns=["Label"])

# Plot and save histogram
fig = graphing.histogram(df, title="")
graphing.save_plot_as_image(fig, file="./2c - unit 03 - skewed_distribution.jpg", format="jpg")


# %%

# 3) a histogram with only True and False (true=250, false=750)
all_true = numpy.full(250, True, dtype=bool)
all_false = numpy.full(750, False, dtype=bool)
data = numpy.hstack((all_true, all_false))
df = pandas.DataFrame(data, columns=["Label"])

# Plot and save histogram
fig = graphing.histogram(df, title="", label_colour="Label")
graphing.save_plot_as_image(fig, file="./2c - unit 03 - boolean_distribution.jpg", format="jpg")

# %%
# 4) a histogram with 'person' (200),  'animal' (400),  and 'tree' (100)
people = numpy.full(200, "person")
animals = numpy.full(400, "animal")
trees = numpy.full(100, "tree")
data = numpy.hstack((people, animals, trees))
df = pandas.DataFrame(data, columns=["Label"])

# Plot and save histogram
fig = graphing.histogram(df, title="", label_colour="Label")
graphing.save_plot_as_image(fig, file="./2c - unit 03 - people etc distribution 1.jpg", format="jpg")

# %%
# 4) a histogram with 'person' (400),  'animal' (200),  'tree' (800), and 'rock' (800)
people = numpy.full(400, "person")
animals = numpy.full(200, "animal")
trees = numpy.full(800, "tree")
rocks = numpy.full(800, "rock")
data = numpy.hstack((people, animals, trees, rocks))
df = pandas.DataFrame(data, columns=["Label"])

# Plot and save histogram
fig = graphing.histogram(df, title="", label_colour="Label")
graphing.save_plot_as_image(fig, file="./2c - unit 03 - people etc distribution 2.jpg", format="jpg")
