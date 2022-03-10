import torch
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import datetime

from typing import Iterable, Union, Tuple, Sequence, Optional
from os import PathLike
import numpy.typing as npt

import logging


Numeric = Union[float, int]

#Some constants for doing visualizations
_IMG_SIZE: Tuple[int, int] = (400, 400)
_WIDTH: int = 100
_HEIGHT: int = 250
_BARRIER_WIDTH: int = 50
_TIMESTAMP_FMT: str = "%Y-%m-%dT%H:%M:%S"


def _rescale(data: Union[Numeric, npt.NDArray], old_range: Tuple[float, float], new_range: Tuple[float, float] = (0.0, 1.0)) -> Union[Numeric, npt.NDArray]:
    '''
    Rescales data linearly from given range to a new range.

    Parameters
    ----------
    data: Numeric | npt.NDArray:
        The data to scale. Can be either a single numeric value or an array of values.
    old_range: Tuple[float, float]
        The old range, where the value(s) are currently located
    new_range: Tuple[float, float], Optional
        The range, where the value will be scaled to. Default (0, 1)
    Returns
    -------
    Numeric | npt.NDArray
        The rescaled data. Will be of same type as the passed in value. 
    '''
    min_val, max_val = min(old_range), max(old_range)
    return ((data - min_val) / (max_val - min_val))* (max(new_range) - min(new_range))


def _create_env_image() -> npt.NDArray:
    '''
    Creates a conceptual image of the maze. Used
    for illustrative purposes

    Returns
    ------
    npt.NDArray
        The created gray-scale image.
    '''

        #Create the 'enviroment' presentation
    base_img = np.ones(_IMG_SIZE)
    mid = _IMG_SIZE[0]//2
    base_img[mid - _WIDTH//2: mid + _WIDTH//2, :_HEIGHT] = 0
    base_img[:_BARRIER_WIDTH, :] = 0
    base_img[_IMG_SIZE[0] - _BARRIER_WIDTH:, :] = 0
    base_img[:, _IMG_SIZE[0] - _BARRIER_WIDTH: ] = 0
    return base_img


def _set_logging_config() -> None:
    '''Sets up the configuration for the logger '''
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)s/%(funcName)s [%(levelname)s] %(message)s",
        datefmt=_TIMESTAMP_FMT,
        handlers= [
            logging.FileHandler("training.log"), 
            logging.StreamHandler()
        ]
    )    


def label_goals(returns: Iterable[float], rmin=0.1, rmax=0.9) -> npt.NDArray[np.int32]:
    '''
    Labels goals based on if they are in the valid range
    
    Parameters
    ----------
    returns: Iterable[float]
        The return values as given by the current policy
    rmin: float, optional
        The lower bound accepted to be in the GOID. Default 0.1
    rmax: float, optional
        The upper bound accepted to be in the GOID. Default 0.9
    
    Returns
    -------
    np.ndarray:
        A numpy array of ints, where 1 means that return value was in range [rmin, rmax]. 
    '''
    return np.array([int(rmin <= r <= rmax) for r in returns], dtype=np.int32)


def sample_tensor(x: torch.Tensor, amount: int) -> torch.Tensor:
    '''
    Sample from a given tensor, see https://stackoverflow.com/a/59461812 for explainer.

    Parameters
    ----------
    x: torch.Tensor:
        The tensor to sample from.
    amount: Int
         The amount of samples to take. 

    Returns
    -------
    torch.Tensor
        A tensor containing sampled data from the input tensor
    '''
    samples = min(len(x), amount)
    idx = torch.randperm(len(x))[:samples]
    return x[idx]

def get_device() -> torch.device:
    ''' Returns the device to use. Uses GPU, if available'''
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_repr(device: Union[str, torch.device]) -> str:
    '''
    Get a Human readable presentation of the used device.
    
    Parameters
    ----------
    device: str | torch.device
        The device, where the presentation is created from
    
    Returns
    -------
    str
        A human readable presentation of the device.
    '''
    device = torch.device(device) if isinstance(device, str) else device
    return torch.cuda.get_device_name(device) if device.type == 'cuda' else "cpu"



def display_agent_and_goals(agent_pos: npt.NDArray, goals: npt.ArrayLike, coord_range: Tuple[float, float], filepath: Optional[Union[str, PathLike]] = None, ax: plt.Axes = None, **kwargs) -> None:
    '''
    Displays the current position of the agent, and the current goals.

    Parameters
    ----------
    agent_pos: npt.NDArray
        The position of the agent. Should be an 2 element array
    goals: npt.NDArray
        The goals. Each element should be a 2-element array 
        presenting the (x,y) coordinate pairs
    coord_range: Tuple[float, float]
        The coordinate range, where the given points are in.
    filepath: Optional[str | PathLike]
        Filepath, where the image will be saved to. If it is None, 
        the image is not saved. Note: To use this option, ax must be None. Default None.
    ax: Optional[plt.Axes]
        The axis where the goals and and agent will be displayed. If it 
        is None, then a new axis will be made. Default None.
    kwargs: named arguments:
        title: str 
            Title for the image
        pos_label: str
            Label for the agent position
        goal_label: str
            Label for the goal(s)
        All other specified option's will be given to the ax.imshow method
    '''
    base_img = _create_env_image()
    fig = None

    if filepath is not None and ax is not None:
        raise ValueError("If ax is specified, the filepath cannot be specified!")
        
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))


    pos_label = kwargs.pop("pos_label", None)
    goal_label = kwargs.pop("goal_label", None)
    title = kwargs.pop("title", None)

    #Plot the actual image
    ax.imshow(base_img, cmap="gray", **kwargs)
    ax.grid(None)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title(title)

    scaled_pos = _rescale(agent_pos, coord_range, (0, _IMG_SIZE[0]))
    scaled_goals = _rescale(goals, coord_range, (0, _IMG_SIZE[0]))

    x, y = scaled_goals if scaled_goals.shape[0] == 2 else scaled_goals.T

    ax.scatter(scaled_pos[0], scaled_pos[1], c="r", s=60, marker='x', label=pos_label)
    ax.scatter(x, y, c="b", s=60, marker="o", label=goal_label)
    ax.invert_yaxis() #Y-axis is inverted in the coordinate systems.

    #If one of the labels was specified, draw it.
    if goal_label is not None or pos_label is not None:
        ax.legend(fontsize=8)

    if fig is not None and filepath is not None:
        fig.savefig(filepath)    
        plt.close()



def line_plot_1d(x: npt.NDArray, y: npt.NDArray, filepath: Union[str, PathLike], label: Optional[str] = None, **kwargs) -> None:
    '''
    Creates a 1d line-plot with standard deviation interval around the plotted line.

    Parameters
    ----------
    x: npt.NDArray
        The x values. Should be an 1D  array
    y: npt.NDArray
        The actual data to plot. Should be an 1D array.
    filepath: str | PathLike
        Path to the location where the figure will be stored.
    label: Optional[str]
        Label for the line. Default None
    kwargs: Named arguments
        title: str
            The title of the plot. Default Results
        xlabel: str
            The x label for the plot. Default x
        ylabel: str
            The y label for the plot. Default y
        alpha: float
            The opacity of the fill. Default 0.2
        Other named arguments will be passed to plot method.
    '''
    
    axis = y.shape.index(max(y.shape))
    std = y.std(axis=axis)

    title = kwargs.pop("title", "Results")
    xlabel = kwargs.pop("xlabel", "x")
    ylabel = kwargs.pop("ylabel", "y")
    alpha = kwargs.pop("alpha", 0.2)

    fig, ax = plt.subplots(1,1,  figsize=(10, 10))
    ax.plot(x, y, label=label, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.fill_between(x, y, y + std, antialiased=True, alpha=alpha)
    ax.fill_between(x, y, y - std, antialiased=True, alpha=alpha)

    if label is not None:
        ax.legend()
    fig.savefig(filepath)
    plt.close(fig)  


def line_plot(x: npt.NDArray, y: npt.NDArray, filepath: Union[str, PathLike], labels: Sequence[str], **kwargs) -> None:
    '''
    Displays the x and y data in a line plot, with standard deviation interval around them. 

    Parameters
    ----------
    x: npt.NDArray
        The x values. Should be 1D array, as all the plottings should share the same x-axis.
    y: npt.NDArray
        The actual data. Can be a 2D array.
    filepath: str | Pathlike
        Path to the location where the figure will be stored.
    labels: Sequence[str]
        A sequence of labels, corresponding to the different y-data values.  
    '''

    #Get the correct axis to calculate the standard deviation on
    axis = y.shape.index(max(y.shape))
    std = y.std(axis=axis)
    
    #Extract some user specified options
    title = kwargs.pop("title", "Results")
    xlabel = kwargs.pop("xlabel", "x")
    ylabel = kwargs.pop("ylabel", "y")
    alpha = kwargs.pop("alpha", 0.2)


    fig, ax = plt.subplots(1,1, figsize=(10,10))
    for i in range(y.shape[0]):
        ax.plot(x, y[i, :], label=labels[i], **kwargs)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.fill_between(x, y[i,:], y[i, :] + std[i], antialiased=True, alpha=alpha)
        ax.fill_between(x, y[i, :], y[i, :] - std[i], antialiased=True, alpha=alpha)
    ax.legend()
    fig.savefig(filepath)
    plt.close(fig)     
    

def timestamp_path(path: Union[str, PathLike]) -> PathLike:
    '''
    Adds a timestamp to a given paths filename (returned by pathlib.Path.name)
    The used timestamp is formatted with ISO 8061 format

    Parameters
    ----------
    path: str | PathLike
        The path, where the timestamp needs to be added
    
    Returns
    -------
    PathLike
        The same path, but with a filename that contains a 
        timestamp.

    '''
    path = pathlib.Path(path) if not isinstance(path, pathlib.PurePath) else path
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") #ISO 8061 format    
    parts = path.name.split(".")
    assert len(parts) > 1, f"Couldn't find a suffix from the given path: {path}"
    newname = f"{parts[0]}{now}.{'.'.join(parts[1:])}"
    return path.parent / newname


def add_to_path(path: Union[str, PathLike], to_add: str) -> PathLike:
    '''
    Adds a string to the end of a filename (returned by pathlib.Path.name). 
    The added text is separated with '_' from the original filename.
    
    Parameters
    ----------
    path: str | PathLike
        The path to modify
    to_add: str
        Any string that should be added to the end of the filename.
    
    Returns
    ------
    PathLike
        A new path-object, where the new string is appended to the end of the 
        original filename.

    '''
    path = pathlib.Path(path) if not isinstance(path, pathlib.PurePath) else path
    parts = path.name.split(".")
    assert len(parts) > 1, f"Couldn't find a suffix from the given path: {path}"
    newname = f"{parts[0]}_{to_add}.{''.join(parts[1:])}"
    return path.parent / newname


def get_logger(name: str) -> logging.Logger:
    '''
    Returns a logger for a given name.

    Parameters
    ----------
    name: str
        The name of the logger, often same as the module.

    Returns
    -------
    logging.Logger
        The created/already existing logger.

    '''
    return logging.getLogger(name)


#Make sure that the logger is initialized
_set_logging_config()