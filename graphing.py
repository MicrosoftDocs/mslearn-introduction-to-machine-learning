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
template.layout = graph_objects.Layout(margin=dict(l=2, r=2, b=2, t=2),
                                    hovermode="closest",
                                    xaxis_showline=True,
                                    xaxis_linewidth=2,
                                    yaxis_showline=True,
                                    yaxis_linewidth=2
                                    )    
                                    
template.data.scatter = [graph_objects.Scatter(marker=dict(opacity=0.8))]

pio.templates["custom_template"] = template
pio.templates.default = "plotly_white+custom_template" #  

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
    '''

    # Automatically pick columns if not specified
    if label_x is None:
        label_x = df.columns[0]
    if label_y is None:
        label_y = df.columns[1]

    # Create human-readable labels
    x_axis_label = label_x.replace('_', ' ')
    y_axis_label = label_y.replace('_', ' ')


    # Create the figure and plot
    fig = px.scatter(df, 
                x=label_x, 
                y=label_y, 
                color=label_colour, 
                labels={
                    label_x:x_axis_label,
                    label_y:y_axis_label,
                    },
                title=title
                )

    # User a marker size inversely proportional to the number of points
    size = int(round(22.0 - 20/(1+exp(-(df.shape[0]/100-2)))))
    print(size)
    fig.update_traces(marker={'size': size})

    # # Remove most whitespace from the edges
    # # Set the background colour
    # fig.update_layout(margin=dict(l=2, r=2, b=2, t=2))

    # # Colour the vertical and horizontal lines 
    # fig.update_xaxes(showline=True, linewidth=AXIS_LINE_WIDTH)
    # fig.update_yaxes(showline=True, linewidth=AXIS_LINE_WIDTH)


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
    '''

    # Automatically pick columns if not specified
    if label_x is None:
        label_x = df.columns[0]
    if label_y is None:
        label_y = df.columns[1]
    if label_z is None:
        label_z = df.columns[2]
    if label_colour is None:
        label_colour = label_z

    # Create human-readable labels
    x_axis_label = label_x.replace('_', ' ')
    y_axis_label = label_y.replace('_', ' ')
    z_axis_label = label_z.replace('_', ' ')

    # Create the figure and plot
    fig = px.scatter_3d(df, 
                x=label_x, 
                y=label_y, 
                z=label_z,
                color=label_colour,
                labels={
                    label_x:x_axis_label,
                    label_y:y_axis_label,
                    label_z:z_axis_label
                    },
                title=title                )

    # Pick a slightly different P.O.V from default
    # this avoids the extremities of the y and x axes
    # being cropped off
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.1)))

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the plot
    return fig
