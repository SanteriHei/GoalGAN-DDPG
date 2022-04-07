import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import copy

from typing import Union, Tuple, Sequence
import numpy.typing as npt



#Note: The implementation presented here is HEAVILY "inspired" 
#by this repo: https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents


class ReplayBuffer:

    '''
    Simple replay buffer implementation for DDPG agents

    Parameters
    ----------
    action_size: int
        The size of an action.
    state_size: int
        The size of a state.
    buffer_size: int
        The maximum size of this buffer.
    batch_size: int
        The size of a batch sampled from this buffer
    device: torch.device
        The device where the sampled tensors should be located in.
    '''
    def __init__(self, action_size: int, state_size: int, buffer_size: int, batch_size: int, device: torch.device) -> None:
        
        # ------ Housekeeping items --------------
        self._len: int = 0
        self._ptr: int = 0
        self._buffer_size: int = buffer_size
        self._batch_size: int = batch_size
        self._rng: np.random.Generator = np.random.default_rng()
        self._device = device

        # ----------- Buffers --------------------
        self._state_buffer: npt.NDArray = np.zeros((buffer_size, state_size), dtype=np.float32)
        self._action_buffer: npt.NDArray = np.zeros((buffer_size, action_size), dtype=np.float32)
        self._reward_buffer: npt.NDArray =  np.zeros((buffer_size,), dtype=np.float32)
        self._done_buffer: npt.NDArray = np.zeros((buffer_size), dtype=np.float32)
        self._next_state_buffer: npt.NDArray = np.zeros((buffer_size, state_size), dtype=np.float32)

    @property
    def batch_size(self) -> int:
        '''Returns the batch size of the buffer'''
        return self._batch_size


    def append(self,  state: npt.NDArray, action: npt.NDArray, reward: float, done: bool,next_state: npt.NDArray) -> None:
        '''
        state: npt.NDArray
            The state where the action was taken from.
        action: npt.NDArray
            The taken action.
        reward: float
            The received reward after the action was taken.
        done: bool
            The status of the task after the action was taken.
        next_state: npt.NDArray
            The state that was reached by taking the action.
        '''
        self._state_buffer[self._ptr] = state
        self._action_buffer[self._ptr] = action 
        self._reward_buffer[self._ptr] = reward
        self._done_buffer[self._ptr] = float(done)
        self._next_state_buffer[self._ptr] = next_state
        
        self._ptr = (self._ptr + 1) % self._buffer_size
        self._len = min(self._len + 1, self._buffer_size)


    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Samples the buffer to create batch size samples.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Returns a tuple of states, actions, rewards, dones and next states from the buffer,
            that are located on the device specified in the constructor.
        '''
        idxs = self._rng.integers(0, self._len, size=self._batch_size)      

        states = torch.from_numpy(self._state_buffer[idxs]).float().to(self._device)
        actions = torch.from_numpy(self._action_buffer[idxs]).float().to(self._device)  
        rewards = torch.from_numpy(self._reward_buffer[idxs]).float().to(self._device)
        dones = torch.from_numpy(self._done_buffer[idxs]).float().to(self._device)
        next_states = torch.from_numpy(self._next_state_buffer[idxs]).float().to(self._device)

        return states, actions, rewards, dones, next_states


    def __len__(self) -> int:
        '''Returns the current lenght of the buffer'''
        return self._len



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
    action_range: int
        The range for the actions. Should be symmetric, 
        i.e. if the range for actions is [-10, 10], set action-range to 10. 
    action_size: int
        The dimension of actions
    fc1_units: int, Optional
        The amount of units in the first hidden layer. Default 256
    fc2_units: int, Optional
        The amount of units in the second hidden layer. Default 256
    use_norm: bool, Optional
        Determines if batch normalization is used between layers. Default False.
    '''
    def __init__(self, state_size: int, action_size: int, action_range: int, fc1_units: int = 256, fc2_units: int = 256, use_norm: bool = False) -> None:
        super(Actor, self).__init__()

        self._action_range: int = action_range

        self._fc1 = nn.Linear(state_size, fc1_units)
        self._norm1 = nn.LayerNorm(fc1_units) if use_norm else None

        self._fc2 = nn.Linear(fc1_units, fc2_units)
        self._norm2 = nn.LayerNorm(fc2_units) if use_norm else None

        self._fc3 = nn.Linear(fc2_units, action_size)
        
        self._use_norm = use_norm
        self.reset()

    def reset(self) -> None:
        '''
        Resets the weights of the model by applying a initalization function
        '''
        self._fc1.weight.data.uniform_(*_hidden_init(self._fc1))
        self._fc2.weight.data.uniform_(*_hidden_init(self._fc2))
        self._fc3.weight.data.uniform_(-3e-3, 3e-3)

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
            The resulting actions, in range of (-action_range, action_range)

        '''

        x = F.relu(self._fc1(x))
        if self._use_norm:
            x = self._norm1(x)
        
        x = F.relu(self._fc2(x))
        if self._use_norm:
            x = self._norm2(x)

        #Tanh returns values between (-1,1). Multiply by the symmetric range to produce correct results
        return torch.tanh(self._fc3(x))*self._action_range



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
    use_norm: bool, Optional
        Determines if batch normalization is used between layers. Default False.
    '''
    def __init__(self, state_size: int, action_size: int, fc1_units: int = 256, fc2_units: int = 256, use_norm: bool = False) -> None:
        super(Critic, self).__init__()
        
        self._fc1 = nn.Linear(state_size, fc1_units)
        self._norm1 = nn.LayerNorm(fc1_units) if use_norm else None

        self._fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self._norm2 = nn.LayerNorm(fc2_units) if use_norm else None
        
        self._fc3 = nn.Linear(fc2_units, 1)
        
        self._use_norm = use_norm


        self.reset()

    def reset(self) -> None:
        '''
        Resets the networks weights by applying initalization function
        '''
        self._fc1.weight.data.uniform_(*_hidden_init(self._fc1))
        self._fc2.weight.data.uniform_(*_hidden_init(self._fc2))
        self._fc3.weight.data.uniform_(-3e-3, 3e-3)

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
        xs = F.relu(self._fc1(state))
        if self._use_norm:
            xs = self._norm1(xs)

        x = torch.cat((xs, action), dim=1)
        x = F.relu(self._fc2(x))
        if self._use_norm:
            xs = self._norm2(x)
        
        return self._fc3(x)


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
        self._theta: float = theta
        self._sigma: float = sigma
        self._sigma_min: float = sigma_min
        self._sigma_decay:float = sigma_decay
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
        dx = self._theta * (self._mu - self._state) + self._sigma * self._rng.standard_normal(size=self._size)
        self._state += dx
        return self._state

