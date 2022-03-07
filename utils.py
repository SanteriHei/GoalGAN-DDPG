
import torch
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import datetime

from typing import Iterable, Union, Tuple, Sequence
from os import PathLike
import numpy.typing as npt


#Some constants for doing visualizations
_IMG_SIZE = (400, 400)
_WIDTH = 100
_HEIGHT = 250
_BARRIER_WIDTH = 50

def _rescale(data: npt.NDArray, range: Tuple[int, int] = (0, 1)) -> npt.NDArray[np.float32]:
    '''
    Rescales data linearly between the given range

    Parameters
    ----------
    data: npt.NDArray:
        The data to scale. The data should be of a numeric type
    range: Tuple[int, int], Optional
        The range, where the values will be scaled to. Default (0, 1)
    
    Returns
    -------
    npt.NDArray
        The rescaled data, as floats.
    '''
    assert abs(max(range)- min(range)) > np.finfo(np.float32).eps , f"Expexted a range that contains different values, but got {range}"
    return (((data - np.min(data))/np.ptp(data)) * (max(range) - min(range))).astype(np.float32)



def label_goals(returns: Iterable[float], rmin=0.1, rmax=0.9) -> npt.NDArray[np.uint32]:
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
    return np.ndarray([int(rmin <= r <= rmax) for r in returns], dtype=np.uint32)


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


def display_goals(goals: npt.ArrayLike, filepath: Union[str, PathLike], rescale: bool = True, **kwargs) -> None:
    '''
    Displays the current goals on a figure, and saves the figure. Note, due to bug with mujoco-py,
    and nvidia drivers, the goals will be displayed over a representive figure of the environment, not over
    the real environment. 
    
    Parameters
    ----------
    goals: npt.Arraylike
        The goals to display. Must have shape of (2, *) or (*, 2), where
        one of the channels represents the x-components and other the y-components 
    
    filepath: Union[str, os.pathlike]
        Path to file where the figure will be saved
    
    rescale: bool, Optional
        Define if the data needs to be rescaled before plotted. Default true.

    kwargs: name arguments
        title: str -> Adds a title to the figure
        label: str -> Adds label to the goals
        All other specified options are passed to the plt.imshow command.
    '''
    goals = np.array(goals) if not isinstance(goals, np.ndarray) else goals

    assert goals.shape[0] == 2 or goals.shape[1] == 2, f"The goals should contain x, y components, now the size was {goals.shape}"
    
    
    #Create the 'enviroment' presentation
    base_img = np.ones(_IMG_SIZE)
    mid = _IMG_SIZE[0]//2
    base_img[mid - _WIDTH//2: mid + _WIDTH//2, :_HEIGHT] = 0
    base_img[:_BARRIER_WIDTH, :] = 0
    base_img[_IMG_SIZE[0] - _BARRIER_WIDTH:, :] = 0
    base_img[:, _IMG_SIZE[0] - _BARRIER_WIDTH: ] = 0
    
    #Plot the actual image
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    title = kwargs.pop("title", None)
    if title:
        ax.set_title(title)

    label = kwargs.pop("label", None)
    ax.imshow(base_img, cmap="gray", **kwargs)
    
    #Rescale the data to match the enviroment.
    if rescale:
        goals = _rescale(goals, (0, 400))
    
    x, y = goals if goals.shape[0] == 2 else goals.T    
    if label:
        ax.scatter(x, y, label=label)
        ax.legend()
    else:
        ax.scatter(x, y)

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