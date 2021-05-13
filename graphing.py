'''
Several no-fuss methods for creating plots 
'''
from typing import Optional, Callable, Union, List
from numpy import exp
from numpy.core.fromnumeric import repeat, shape
import pandas
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as graph_objects

# Set the default theme 
template =  graph_objects.layout.Template() 
template.layout = graph_objects.Layout(
                                    # border width
                                    margin=dict(l=2, r=2, b=2, t=2),
                                    # Interaction
                                    hovermode="closest",
                                    # axes
                                    xaxis_showline=True,
                                    xaxis_linewidth=2,
                                    yaxis_showline=True,
                                    yaxis_linewidth=2,
                                    # Pick a slightly different P.O.V from default
                                    # this avoids the extremities of the y and x axes
                                    # being cropped off
                                    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.1))
                                    )    
                                    
template.data.scatter = [graph_objects.Scatter(marker=dict(opacity=0.8))]
template.data.scatter3d = [graph_objects.Scatter3d(marker=dict(opacity=0.8))]


pio.templates["custom_template"] = template
pio.templates.default = "plotly_white+custom_template"

def _to_human_readable(text:str):
    return text.replace("_", " ")

def _prepare_labels(df:pandas.DataFrame, labels:List[Optional[str]]):
    '''
    Ensures labels are human readable. 
    Automatically picks data if labels not provided explicitly
    '''

    human_readable = {}

    for i in range(len(labels)):
        lab = labels[i]
        if lab is None:
            lab = df.columns[i]
        labels[i] = lab

        # make human-readable
        human_readable[lab] = _to_human_readable(lab)
    
    return labels, human_readable


def scatter_2D(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_colour:Optional[str]=None,
                title=None, 
                show:bool=False,
                trendline:Union[Callable,List[Callable],None]=None):
    '''
    Creates a 3D scatter plot and shows it. Returns the figure for that scatter.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: The label to extract from df to plot on the x axis. Defaults to df.columns[0]
    label_y: The label to extract from df to plot on the y axis. Defaults to df.columns[1]
    label_colour: The label to extract from df to colour points by
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    trendline:  A function that accepts X and returns Y

    '''

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y])

    # Create the figure and plot
    fig = px.scatter(df, 
                x=selected_columns[0], 
                y=selected_columns[1], 
                color=label_colour, 
                labels=axis_labels,
                title=title
                )

    # User a marker size inversely proportional to the number of points
    size = int(round(22.0 - 20/(1+exp(-(df.shape[0]/100-2)))))
    fig.update_traces(marker={'size': size})

    # Create trendlines
    if isinstance(trendline, Callable):
        trendline = [trendline]
    x_min = min(df[selected_columns[0]])
    x_max = max(df[selected_columns[0]])
    for t in trendline:
        y_min = t(x_min)
        y_max = t(x_max)
        fig.add_shape(
            type='line',
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max)

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the plot
    return fig


def scatter_3D(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_z:Optional[str]=None, 
                label_colour:Optional[str]=None,
                title=None, 
                show:bool=False):
    '''
    Creates a 3D scatter plot and shows it. Returns the figure for that scatter.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: The label to extract from df to plot on the x axis. Defaults to df.columns[0]
    label_y: The label to extract from df to plot on the y axis. Defaults to df.columns[1]
    label_z: The label to extract from df to plot on the z axis. Defaults to df.columns[2]
    label_colour: The label to extract from df to colour points by. Defaults to label_x
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    '''

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_z])

    if label_colour is None:
        # Colour by the Z dimension
        label_colour = selected_columns[2]
    else:
        axis_labels[label_colour] = _to_human_readable(label_colour)

    # Create the figure and plot
    fig = px.scatter_3d(df, 
                x=selected_columns[0], 
                y=selected_columns[1], 
                z=selected_columns[2],
                color=label_colour,
                labels=axis_labels,
                title=title)


    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the plot
    return fig
