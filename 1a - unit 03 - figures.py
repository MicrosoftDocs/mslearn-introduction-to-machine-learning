'''
This creates figures for content. It is not intended to act as learning material
'''
import graphing
import pandas
import plotly.io


# Example of a straight line
df = pandas.DataFrame(dict(
    age_in_years=[0,8],
    core_temperature=[37.0, 41.0]
))

fig = graphing.scatter_2D(df, title="Relationship between age and body temperature", trendline=lambda x: x*0.5 + 37)
fig.update_traces(marker={'opacity': 0})

plotly.io.write_image(fig, f"1a - unit 03 - figure 1.svg")


# Example of a straight line with data
df = pandas.DataFrame(dict(
    age_in_years=[1,8, 4, 5],
    core_temperature=[38.0, 40.0, 38, 41]
))


fig = graphing.scatter_2D(df, title="Relationship between age and body temperature", trendline=lambda x: x*0.5 + 37)

fig.add_shape(type="line", x0=5, y0=39.5, x1=5, y1=41)
plotly.io.write_image(fig, f"1a - unit 03 - figure 2.svg")

fig.add_shape(type="line", x0=0.5, y0=39.5, x1=5, y1=39.5, line_dash='dash')
fig.add_shape(type="line", x0=0.5, y0=41, x1=5, y1=41, line_dash='dash')
plotly.io.write_image(fig, f"1a - unit 03 - figure 3.svg")

