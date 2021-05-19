'''This file contains code generating graphs for Module 0c Unit 06. It is not intended as a learning resource.'''

from typing import List
import pandas
import graphing # Import to set style info
import plotly.io
import plotly.express as px
import statsmodels.formula.api as smf


dataset = pandas.read_csv('Data/titanic.csv', index_col=False, sep=",", header=0)

def make_plot(order:List[str], show:bool=False):
    '''
    Creates a plot with Embarked port treated as integers

    order: Order of variables. For example ['C','S','Q'] would encode C as 0, S as 1 and Q as 2 
    show: Whether to show the generated graph
    '''

    # Create a column containing ports converted to integers
    dataset["PortAsNumber"] = dataset.Embarked.replace({order[0]:0, order[1]:1, order[2]:2})

    # Begin making a figure
    fig = px.strip(dataset, x="PortAsNumber", y="Pclass", width=400)

    # Replace numbers on axis with human readable values
    fig.update_xaxes(tickvals=[0,1,2], ticktext=order)

    # Perform fit a line to our data using simple linear regression
    model = smf.ols(formula = "Pclass ~ PortAsNumber", data = dataset).fit()
    fig.add_shape(type="line", 
                    x0=0, 
                    x1=2, 
                    y0=model.params[0], 
                    y1=(model.params[1]*2 + model.params[0]))

    # Export SVG
    plotly.io.write_image(fig, f"0c - unit 06 - figure {''.join(order)}.svg")

    if show:
        fig.show()
    
    # Clean up
    del dataset["PortAsNumber"]


make_plot(["C","S","Q"])
make_plot(["S","C","Q"])