'''
This creates figures for content. It is not intended to act as learning material
'''
from numpy.core.function_base import linspace
import graphing
import pandas
import plotly.io
import numpy
import numpy.random

df = pandas.DataFrame(dict(
    x=[-10,10],
    y=[600, -2500]
))



df = pandas.DataFrame(dict(
    x=[-10,5],
    y=[600, -1000]
))

# Example of a straight line
fig = graphing.scatter_2D(df, title="Two parameter (linear)", trendline=lambda x: x*40 + 100)
fig.update_traces(marker={'opacity': 0})
plotly.io.write_image(fig, f"1a - unit 06 - figure 1.svg")

# Example of a quadratic
fig = graphing.scatter_2D(df, title="Three parameter polynomial", trendline=lambda x: -10*x**2 + x*40 + 100)
fig.update_traces(marker={'opacity': 0})
plotly.io.write_image(fig, f"1a - unit 06 - figure 2.svg")


# Example of a cubic
fig = graphing.scatter_2D(df, title="Four parameter polynomial", trendline=lambda x: -1.5*x**3 + -10*x**2 + x*40 + 100)
fig.update_traces(marker={'opacity': 0})
plotly.io.write_image(fig, f"1a - unit 06 - figure 3.svg")



# Three curve comparison
curve_log = numpy.log10
curve_poly = lambda x: -0.0002*x**2 + x*0.025 + 0
curve_sigmoid = lambda x: 1/(1+numpy.exp(-(x-50)/10))

df = pandas.DataFrame(dict(
    x=numpy.linspace(1,100, 100),
    y=numpy.linspace(0,2, 100)
))


fig = graphing.scatter_2D(df, title="Example Curves", trendline=[curve_log, curve_poly, curve_sigmoid])
fig.update_traces(marker={'opacity': 0})
plotly.io.write_image(fig, f"1a - unit 06 - figure 4.svg")


# exponential
curve_log = numpy.log10
curve_poly = lambda x: 0.0005*x**2 + x*-0.025 + 0.25
curve_sigmoid = lambda x: 1/(1+numpy.exp(-(x-50)/10))
x = numpy.linspace(1,100, 100)
y = curve_poly(x) + numpy.random.rand(x.shape[0])*0.25-0.125
df = pandas.DataFrame(dict(
    x=x,
    y=y
))


fig = graphing.scatter_2D(df, size_multiplier=0.5, trendline=[curve_log, curve_poly, curve_sigmoid])
# fig.update_traces(marker={'opacity': 0})
plotly.io.write_image(fig, f"1a - unit 06 - figure 5.svg")



# exponential out of control
curve_poly = lambda x: (-1.5/150*37.5*x**3 + -10/150*37.5*x**2 + x*40/150*37.5) + 90
x = numpy.linspace(-10,2, 20) + numpy.random.rand(20)*0.5-0.25
y = curve_poly(x) + numpy.random.rand(x.shape[0])*10-5
df = pandas.DataFrame(dict(
    x=x,
    y=y
))


fig = graphing.scatter_2D(df, x_range=[-10,10], size_multiplier=0.5, trendline=[curve_poly])
# fig.update_traces(marker={'opacity': 0})
plotly.io.write_image(fig, f"1a - unit 06 - figure 6.svg")


# # Example of a straight line with data
# df = pandas.DataFrame(dict(
#     age_in_years=[1,8, 4, 5],
#     core_temperature=[38.0, 40.0, 38, 41]
# ))


# fig = graphing.scatter_2D(df, title="Relationship between age and body temperature", trendline=lambda x: x*0.5 + 37)

# fig.add_shape(type="line", x0=5, y0=39.5, x1=5, y1=41)
# plotly.io.write_image(fig, f"1a - unit 03 - figure 2.svg")

# fig.add_shape(type="line", x0=0.5, y0=39.5, x1=5, y1=39.5, line_dash='dash')
# fig.add_shape(type="line", x0=0.5, y0=41, x1=5, y1=41, line_dash='dash')
# plotly.io.write_image(fig, f"1a - unit 03 - figure 3.svg")

