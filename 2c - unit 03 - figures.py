#%%

"""
This creates figures for content. It is not intended to act as learning material
"""
from numpy.core.function_base import linspace
import graphing
import pandas
import plotly.io
import numpy
import numpy.random
from scipy.stats import norm


# We need the following **histograms**. When the script is run, it should save 
# each as a jpg. 
# Leave the size as default. Don't commit the images to git:
# 1) Random data that are normally distributed. Mean 0, std dev 1
# 2) a skewed normal distribution - similar to the above distribution but
#  where the positive (right) tail is much longer than the left. The mean should still be 0
# 3) a histogram with only True and False (true=250, false=750)
# 4) a histogram with 'banana' (200),  'apple' (400),  and 'cherry' (100)


# 1: Randomly distributed data
data = norm.rvs(0, 0.1, size=1000)
df = pandas.DataFrame(data, columns=["Data"])

# Plot the histogram.
fig = graphing.histogram(df, title="Title",  show=True)
# Save image
graphing.save_plot_as_image(fig, file="./normal_distribution.jpg", format="jpg")




