from importlib.resources import path
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import datetime

from typing import Iterable, Union, Tuple, Optional, Mapping
from os import PathLike
import numpy.typing as npt

import logging


Numeric = Union[float, int]

#Some constants for doing visualizations
_IMG_SIZE: Tuple[int, int] = (400, 400)
_WIDTH: int = 100
_HEIGHT: int = 250
_BARRIER_WIDTH: int = 50
_TIMESTAMP_FMT: str = "%m-%d %H:%M:%S"
_COLORS: Mapping[str, str] = {
    "low": "r",
    "goid": "b",
    "high": "g"
}


_LOG_DIR: str= f"runs/{datetime.datetime.now().strftime('%m-%dT%H:%M')}"


def compined_shape(length: int, shape: Optional[Tuple]=None) -> Tuple:
    '''
    Calculates the computed shape and returns it as a single tuple.

    Parameters
    ----------
    length: int
        The lenght of the array
    shape: Optional[Tuple]
        The shape of the object. Default None.
    
    Returns
    -------
    Tuple:
        Returns the combined shape. If shape is none, returns (lenght, ) tuple,
        otherwise combines the length and shape to a single tuple
    ''' 
    if shape is None:
        return (length,)
    return (length, shape) if np.ndim(shape) == 0 else (length, *shape)

def rescale(data: Union[Numeric, npt.NDArray], old_range: Tuple[float, float], new_range: Tuple[float, float] = (0.0, 1.0)) -> Union[Numeric, npt.NDArray]:
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
    old_min, old_max = min(old_range), max(old_range)
    new_min, new_max = min(new_range), max(new_range)
    return (((data - old_min) / (old_max - old_min))* (new_max - new_min)) + new_min 


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


def set_logging_config() -> None:
    '''Sets up the configuration for the logger '''
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(module)-10s:%(funcName)-25s %(levelname)-8s %(message)s",
        datefmt=_TIMESTAMP_FMT,
        filename="training.log",
        filemode="w"
    )    

    console= logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(module)-10s:%(funcName)-25s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    #Hack to stop matplotlib from polluting the log-file
    logging.getLogger('matplotlib.font_manager').disabled = True



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

def is_symmetric_range(range: Tuple[float, float]) -> bool:
    '''Returns true if the given range is symmetric (i.e. from -x to x'''
    return abs(range[0]) == abs(range[1])

def display_agent_and_goals(
    agent_pos: npt.NDArray, goals: npt.ArrayLike, returns: npt.ArrayLike, 
    coord_range: Tuple[float, float], rmin: float, rmax: float, 
    filepath: Optional[Union[str, PathLike]] = None, ax: plt.Axes = None, **kwargs
) -> None:
    '''
    Displays the current position of the agent, and the current goals.

    Parameters
    ----------
    agent_pos: npt.NDArray
        The position of the agent. Should be an 2 element array
    goals: npt.NDArray
        The goals. Each element should be a 2-element array 
        presenting the (x,y) coordinate pairs
    returns: npt.NDArray
        The evaluation values of the goals
    coord_range: Tuple[float, float]
        The coordinate range, where the given points are in.
    rmin: float
        The minimum evaluation value that is counted as feasible goal.
    rmax: float
        The maximum evalution value that is counted as feasible goal.
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
        All other specified option's will be given to the ax.imshow method
    '''
    base_img = _create_env_image()
    fig = None

    if filepath is not None and ax is not None:
        raise ValueError("If ax is specified, the filepath cannot be specified!")
        
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))


    pos_label = kwargs.pop("pos_label", None)
    title = kwargs.pop("title", None)

    #Plot the actual image
    ax.imshow(base_img, cmap="gray", **kwargs)
    ax.grid(None)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title(title)

    scaled_pos = rescale(agent_pos, coord_range, (0, _IMG_SIZE[0]-1))
    scaled_goals = rescale(goals, coord_range, (0, _IMG_SIZE[0]-1))

    scaled_goals = scaled_goals if scaled_goals.shape[0] == 2 else scaled_goals.T

    ax.scatter(scaled_pos[0], scaled_pos[1], c="m", s=60, marker='x', label=pos_label)

    #Separate the easy, feasible and difficult goals
    low_res = scaled_goals[:, returns < rmin]    
    high_res = scaled_goals[:, returns > rmax]
    goid = scaled_goals[:, ((returns >= rmin) & (returns <= rmax))]

    ax.scatter(low_res[0], low_res[0], c=_COLORS["low"], s=60, marker="o", label=r"$R < R_{min}$")
    ax.scatter(goid[0], goid[1], c=_COLORS["goid"], s=60, marker="o", label=r"$R_{min} \leq R \leq R_{max}$")
    ax.scatter(high_res[0], high_res[1], c=_COLORS["high"], s=60, marker="o", label=r"$R > R_{max}$")
    
    ax.invert_yaxis() #Y-axis is inverted in the coordinate systems.
    
    ax.legend(fontsize=8, loc="upper right")

    if fig is not None and filepath is not None:
        filepath = filepath if isinstance(filepath, pathlib.PurePath) else pathlib.Path(filepath)
        
        #If the parent directories, create them.
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix != '.svg': 
            warnings.warn(f"Note: Savign as .svg is recommended (using {filepath.suffix} now)")

        fig.savefig(filepath)    
        plt.close()


def line_plot(x: npt.NDArray, y: npt.NDArray, add_std: bool = False, filepath: Optional[Union[PathLike, str]] = None, close_fig: bool = False, **kwargs) -> Optional[plt.Figure]:
    '''
    Displays the x and y data in a line plot. 

    Parameters
    ----------
    x: npt.NDArray
        The x values. Should be 1D array, as all the plottings should share the same x-axis.
    y: npt.NDArray
        The actual data. should be 1D array.
    add_std: bool, Optional
        If set to true, the standard deviation of the data will be added to the plot. Default False
    filepath: Optional[PathLike | str]
        Path to a file, where the figure will be saved to. If None, the figure will not be saved. Default None.
    close_fig: bool, Optional
        If set to true, the handle to the figure is closed. See Returns. Default False.

    kwargs: Named arguments
        title: str
            The title of the plot. Default Results
        xlabel: str
            The x label for the plot. Default x
        ylabel: str
            The y label for the plot. Default y
        label: str
            The label for the data. Default None.
        figsize: Tuple[int, int]
            The size of the created figure. Default (10, 10)
        alpha: float
            The opacity of a fill. Used only if add_std is set to True. Default 0.2
        Other named arguments will be passed to plot method.
    
    Returns
    -------
    Optional[plt.Figure]:
        Returns handle to the plotted figure, if close_fig is set to False, otherwise returns None.
    '''
    title = kwargs.pop("title", "results")
    xlabel = kwargs.pop("xlabel", "x")
    ylabel = kwargs.pop("ylabel", "y")
    label = kwargs.pop("label", None)
    figsize = kwargs.pop("figsize", (10, 10))
    alpha = kwargs.pop("alpha", 0.2)
    grid = kwargs.pop("grid", None)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.plot(x, y, label=label, **kwargs)
    if add_std:
        axis = y.shape.index(max(y.shape))
        std = y.std(axis=axis)
        ax.fill_between(x, y, y + std, antialiased=True, alpha=alpha)
        ax.fill_between(x, y, y - std, antialiased=True, alpha=alpha)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)

    if label is not None:
        ax.legend()

    if filepath is not None:
        fpath = filepath if isinstance(filepath, pathlib.PurePath) else pathlib.Path(filepath)
        if not fpath.parent.exists():
            fpath.parent.mkdir(parents=True, exist_ok=True)
        
        if fpath.suffix != '.svg': 
            warnings.warn(f"Note: Savign as .svg is recommended (using {filepath.suffix} now)")
        fig.savefig(fpath)
    
    if close_fig:
        plt.close(fig)
        return None
    return fig

 
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


def get_writer() -> SummaryWriter: 
    ''' Returns a tensorboard summary writer with correct logging dir'''
    return SummaryWriter(_LOG_DIR)