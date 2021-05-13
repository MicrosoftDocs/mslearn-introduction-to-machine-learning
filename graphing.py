'''
Several no-fuss methods for creating plots 
'''
from typing import Optional, Callable, Union, List
from numpy import exp
import numpy
from numpy.core.fromnumeric import repeat, shape
import pandas
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as graph_objects

# Set the default theme 
template =  graph_objects.layout.Template() 
template.layout = graph_objects.Layout(
                                    title_x=0.5,
                                    # border width
                                    margin=dict(l=2, r=2, b=2, t=30),
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
template.data.histogram = [graph_objects.Histogram()]


pio.templates["custom_template"] = template
pio.templates.default = "plotly_white+custom_template"


def _to_human_readable(text:str):
    '''
    Converts a label into a human readable form
    '''
    return text.replace("_", " ")


def _prepare_labels(df:pandas.DataFrame, labels:List[Optional[str]], replace_nones:bool=True):
    '''
    Ensures labels are human readable. 
    Automatically picks data if labels not provided explicitly
    '''

    human_readable = {}

    for i in range(len(labels)):
        lab = labels[i]
        if replace_nones and (lab is None):
            lab = df.columns[i]
            labels[i] = lab

        # make human-readable
        if lab is not None:
            human_readable[lab] = _to_human_readable(lab)
    
    return labels, human_readable


def histogram(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_colour:Optional[str]=None,
                nbins:Optional[int]=None,
                title=None, 
                show:bool=False):
    '''
    Creates a 2D histogram and optionally shows it. Returns the figure for that histogram.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: What to bin by. Defaults to df.columns[0]
    label_y: If provided, the sum of these numbers becomes the y axis. Defaults to count of label_x
    label_colour: If provided, creates a stacked histogram, splitting each bar by this column
    title: Plot title
    nbins: the number of bins to show. None for automatic
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 

    '''

    # Automatically pick columns if not specified
    if label_x is None:
        label_x = df.columns[0]
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_colour], replace_nones=False)

    print(selected_columns)

    fig = px.histogram(df,
                        x=selected_columns[0],
                        y=selected_columns[1],
                        nbins=nbins,
                        color=label_colour,
                        labels=axis_labels,
                        title=title
                        )
    
    fig.update_traces(marker=dict(line=dict(width=1)))


    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig


def scatter_2D(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_colour:Optional[str]=None,
                title=None, 
                show:bool=False,
                trendline:Union[Callable,List[Callable],None]=None):
    '''
    Creates a 3D scatter plot and optionally shows it. Returns the figure for that scatter.

    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called

    df: The data
    label_x: The label to extract from df to plot on the x axis. Defaults to df.columns[0]
    label_y: The label to extract from df to plot on the y axis. Defaults to df.columns[1]
    label_colour: The label to extract from df to colour points by
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    trendline:  A function that accepts X (a numpy array) and returns Y (an iterable)

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
    size = int(round(22.0 - 19/(1+exp(-(df.shape[0]/100-2)))))
    fig.update_traces(marker={'size': size})

    # Create trendlines
    if trendline is not None:
        if isinstance(trendline, Callable):
            trendline = [trendline]
        x_min = min(df[selected_columns[0]])
        x_max = max(df[selected_columns[0]])
        evaluate_for = numpy.linspace(x_min, x_max, num=200)
        shapes = []
        for t in trendline:
            y_vals = t(evaluate_for)
            path = "M" + " L ".join([str(c[0]) + " " + str(c[1]) for c in zip(evaluate_for,y_vals)])
            shapes.append(dict(
                                type="path",
                                path=path,
                                line_color="Crimson",
                            )
                        )
        
        fig.update_layout(shapes=shapes)

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig


def scatter_3D(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_z:Optional[str]=None, 
                label_colour:Optional[str]=None,
                title=None, 
                show:bool=False):
    '''
    Creates a 3D scatter plot and optionally shows it. Returns the figure for that scatter.

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
    
    # return the figure
    return fig
