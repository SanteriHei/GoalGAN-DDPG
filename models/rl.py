import torch
import torch.nn.functional as F
from torch import nn
import copy
import numpy as np
import random

from collections import deque, namedtuple

from typing import Deque, Union, Tuple, Sequence
import numpy.typing as npt



#Note: The implementation presented here is HEAVILY "inspired" 
#by this repo: https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class MemoryBuffer:
    '''
    Creates a MemoryBuffer to be used as the experience replay in DDPG

    Attributes
    ----------
    _buffer: deque
        A deque, used as the memory buffer.
    
    Parameters
    ----------
    action_size: int
        The size of the action space
    buffer_size: int
        The maximum size of the memory buffer
    batch_size: int
        The amount of samples to return when sampling the memory
    device: torch.device
        The device, where the tensors should be moved to.

    '''
    
    def __init__(self, action_size: int, buffer_size: int, batch_size: int, device: torch.device) -> None:
        self._action_size: int = action_size
        self._buffer: Deque[Experience] = deque(maxlen=buffer_size)
        self._batch_size: int = batch_size
        self._device: torch.device = device

    @property
    def batch_size(self) -> int:
        '''Returns the batch size of the buffer'''
        return self._batch_size

    def append(self, experience: Experience) -> None:
        '''
        Adds new experince to the buffer

        Parameters
        ----------
        experience: Experience
            The experience to add. Contains the current state, action, reward, the next state, and information
            if the task is done
         
        Returns
        -------
            None    
        '''
        self._buffer.append(experience) 

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Sample the experience relay/Memory buffer

        Returns
        -------
        Tuple[torch.Tensor]
            Returns a tuple of states, actions, rewards, next_states and dones from the 
            experience replay, where each value is a tensor with shape (batch_size, )
        '''
        experiences = random.sample(self._buffer, k=self._batch_size)
        action_size = experiences[0].action.shape[0]
        state_size = experiences[0].state.shape[0]


        #Create memory for the samples
        states = torch.zeros(self._batch_size, state_size)
        actions = torch.zeros(self._batch_size, action_size)
        rewards = torch.zeros(self._batch_size)
        next_states = torch.zeros(self._batch_size, state_size)
        dones = torch.zeros(self._batch_size)
        
        for i,e in enumerate(experiences):
            states[i] = torch.from_numpy(e.state)
            actions[i] = torch.from_numpy(e.action)
            rewards[i] = e.reward
            next_states[i] = torch.from_numpy(e.next_state)
            dones[i] = int(e.done)

        states = states.float().to(self._device)
        actions = actions.float().to(self._device)
        rewards = rewards.float().to(self._device)
        next_states = next_states.float().to(self._device)
        dones = dones.float().to(self._device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        '''
        Returns the current length of the buffer
        '''
        return len(self._buffer)


def _hidden_init(layer: nn.Module) -> Tuple[float, float]:
    '''
    Creates range for weight initialization of a layer

    Parameters
    ----------
    layer: nn.Module
        The layer, which will be initialized
    
    Returns
    -------
    Tuple[float, float]
        A (-limit, limit) tuple, which defines a range for the numbers to reside in-

    '''
    f_in = layer.weight.data.size()[0]
    lim = 1./np.sqrt(f_in)
    return (-lim, lim)



class Actor(nn.Module):
    '''
    Actor (policy) model, that maps the states to actions.
    Uses a simple 3 layer linear network model, where the last layer uses tanh activation.

    Attributes
    ----------
    layers: nn.ModuleList
        Contains the layers used in the Actor model. 

    Parameters
    ----------
    state_size: int
        The dimension of states
    action_size: int
        The dimension of actions
    fc1_units: int, Optional
        The amount of units in the first hidden layer. Default 256
    fc2_units: int, Optional
        The amount of units in the second hidden layer. Default 256
    '''
    def __init__(self, state_size: int, action_size: int, fc1_units: int = 256, fc2_units: int = 256) -> None:
        super(Actor, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        ])
        self.reset()

    def reset(self) -> None:
        '''
        Resets the weights of the model by applying a initalization function
        '''
        self.layers[0].weight.data.uniform_(*_hidden_init(self.layers[0]))
        self.layers[2].weight.data.uniform_(*_hidden_init(self.layers[2]))
        self.layers[4].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies the forward pass of the network, which maps states to actions

        Parameters
        ----------
        x: torch.Tensor
            The current states
        
        Returns
        -------
        torch.Tensor
            The resulting actions

        '''
        for layer in self.layers:
            x = layer(x)
        return x
    

class Critic(nn.Module):
    '''
    Defines a Critic (value) model, that maps (state, action) tuples to Q-values.
    Uses a simple 3 layer linear network model.
    
    Parameters
    ----------
    state_size: int
        The dimension of the states
    action_size: int
        The dimension of the actions
    fsc1_units: int, Optional
        The amount of units in the first hidden layer. Default 256
    fsc2_units: int, Optional
        The amout of units in the second hidden layer. Default 256
    '''
    def __init__(self, state_size: int, action_size: int, fsc1_units: int = 256, fsc2_units: int = 256) -> None:
        super(Critic, self).__init__()
        self._fsc1 = nn.Linear(state_size, fsc1_units)
        self._fsc2 = nn.Linear(fsc1_units + action_size, fsc2_units)
        self._fsc3 = nn.Linear(fsc2_units, 1)
        self.reset()

    def reset(self) -> None:
        '''
        Resets the networks weights by applying initalization function
        '''
        self._fsc1.weight.data.uniform_(*_hidden_init(self._fsc1))
        self._fsc2.weight.data.uniform_(*_hidden_init(self._fsc2))
        self._fsc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Applies the forward pass to the network, which maps the (state, action) pairs to Q-values

        Parameters
        ----------
        state: torch.Tensor
            The current state
        action: torch.Tensor
            The current action
        
        Returns
        -------
        torch.Tensor
            The Q-value.
        '''
        xs = F.relu(self._fsc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self._fsc2(x))
        return self._fsc3(x)


class OUNoise:
    '''
    Implements the Ornstein-Uhlenbeck process. See https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    for quick introduction. The explanation of the parameters can also be found there.

    Parameters
    ----------
    size: int | Tuple
        The size of the sampled noise. Can be a integer (1d array), or Tuple of integers (for tensors) 
    mu: float
    theta: float
    sigma: float
    sigma_min: float
    sigma_defay: float
    seed: int | Sequence[int]
        A seed values(s) for the used random number generator.
    '''
    def __init__(
        self, size: Union[int, Tuple], mu: float = 0.0, theta: float = 0.15, sigma: float = 0.15, 
        sigma_min: float = 0.05, sigma_decay: float = 0.975, seed: Union[int, Sequence[int]] = None
    ) -> None: 
        self._mu = mu * np.ones(size)
        self._theta = theta
        self._sigma = sigma
        self._sigma_min = sigma_min
        self._sigma_decay = sigma_decay
        self._size = size
        self._state = None
        self._rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        self.reset()    

    def reset(self) -> None:
        '''
        Resets the internal state of the process back to mean, and reduces the sigma 
        to it's minimum value 
        '''
        self._state = copy.copy(self._mu)
        self._sigma = max(self._sigma_min, self._sigma*self._sigma_decay)
        

    def sample(self) -> npt.NDArray:
        '''
        Applies the OU process, and samples the random number generator to produce
        sample from this process

        Returns
        -------
        np.ndarray
            The sample from the distribution.
        '''
        assert self._state is not None, "Trying to call sample before setting the state (can be set by calling reset)"
        dx = self._theta * (self._mu - self._state) + self._sigma * self._rng.standard_normal(self._size)
        self._state += dx
        return self._state

