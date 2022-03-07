
from models.rl import Actor, Critic, OUNoise, MemoryBuffer, Experience
from utils import timestamp_path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Tuple, Union
from os import PathLike

import pathlib
import warnings
from dataclasses import dataclass

#Note: The implementation presented here is HEAVILY "inspired" 
#by this repo: https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents

_BACKUP_PATH = pathlib.Path("checkpoints/backups/ddpg.tar")
_NOT_SET: int = -1

@dataclass
class DDPGConfig:
    '''
    Defines configuration parameters for the DDPGAgent

    Attributes
    ----------
    action_size: int | Tuple[int]
        Defines the size of each action. The action MUST have 1-dimensional size, 
        i.e. the tuple defining it's size has only 1 element.
    state_size: int | Tuple[int]
        Defines the size of each state. The state MUST have 1-dimensional size,
        i.e. the tuple defining it's size has only 1 element.
    actor_lr: float, Optional
        The learning rate used with the actor's optimizer. Default 0.0001
    critic_lr: float, Optional
        The learning rate used with the critic's optimizer. Default 0.0001
    weight_decay: float, Optional
        The weigth decay used with the actors and critic's optimizer. Default 0.0 
        See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html for exact meaning of this parameter
    tau: float, Optional
        Interpolation parameter used when doing a soft update to the network. Default 0.001
    gammma: float, Optional
        Discount factor used in the value function of the agent. Default 0.001
    buffer_size: int, Optional
        The maximum size of a memory buffer. Default 100 000
    batch_size: int, Optional
        The size of the sampled batch from the memory buffer. Default 128
    '''
    action_size: Union[int, Tuple[int]] = _NOT_SET
    state_size: Union[int, Tuple[int]] = _NOT_SET
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    weight_decay: float = 0.0
    tau: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = int(1e5)
    batch_size: int = 128


class DDPGAgent:    
    '''
    Defines a DDPG (Deep deterministic policy gradient) agent. This agent has a actor-critic model, where
    the actor computes the (possible continuous) actions directly, rather than a probability distribution
    over the all actions.
    
    Parameters
    ----------
    config: DDPGConfig
        A configuration object defining the configuration of the network.
        See the docstring of DDPGConfig for more information about possible parameters.
    device: Union[str, torch.device]
        The device, where the networks should lie. Can be either a string representing the 
        device, or an actual torch.device
    '''

    def __init__(self, config: DDPGConfig, device: Union[str, torch.device]) -> None:
        
        self._state_size: int = self._check_dim_and_unwrap(config.state_size)
        self._action_size: int = self._check_dim_and_unwrap(config.action_size)
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device

        assert self._state_size != _NOT_SET, f"No state size was specified!"
        assert self._action_size != _NOT_SET, f"No action size was specified!"

        #Actor network
        self._actor_local = Actor(self._state_size, self._action_size).to(self._device)
        self._actor_target = Actor(self._state_size, self._action_size).to(self._device)
        self._actor_optim = optim.Adam(
            self._actor_local.parameters(), 
            lr=config.actor_lr, weight_decay=config.weight_decay
        )

        #Critic network
        self._critic_local = Critic(self._state_size, self._action_size).to(self._device)
        self._critic_target = Critic(self._state_size, self._action_size).to(self._device)
        self._critic_optim = optim.Adam(
            self._critic_local.parameters(),
            lr=config.critic_lr, weight_decay=config.weight_decay
        )

        self._noise = OUNoise((1, self._action_size), seed=None)
        self._buffer = MemoryBuffer(self._action_size, config.buffer_size, config.batch_size, self._device)
        
        self._gamma: float = config.gamma
        self._tau: float = config.tau


    def _check_dim_and_unwrap(value: Union[int, Tuple[int]]) -> int:
        '''
        Unwraps the possible tuple values to int's and checks that their dimensionality
        is 1. 

        Parameters
        ---------
        value: int | Tuple[int]
            The value to unwrap
        
        Returns
        -------
        int
            The original value if it was int, otherwise the value inside of the tuple.

        '''
        if isinstance(value, int):
            return value

        assert len(value) == 1, f"Expected a 1D size, but got {value}, with ndim {len(value)}"
        return value[0]

    def step(self, experience: Experience) -> None:
        '''
        Takes a step with the action, and updates the replay buffer. If enough data is 
        saved, the agent will be trained

        Parameters
        ----------
        experience: Experience
            An experince consisting of state, action, reward, the next state, and information
            about if the task is completed after the action.
        
        Returns
        -------
        None
        '''
        self._buffer.append(experience)

        if len(self._buffer) > self._buffer.batch_size:
            experiences = self._buffer.sample()
            self.learn(experiences)


    def act(self, state: torch.Tensor, use_noise: bool = True, range: Tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
        '''
        Uses the current policy to create actions for the given state

        Parameters
        ----------
        state: torch.Tensor
            The current state.
        use_noise: bool, Optional
            Controls if additional noise is added to the 
            generated actions. Default True.
        range: Tuple[int, int], Optional
            Defines the range, where the values should be clipped to.
        Returns
        -------
        torch.tensor
            The generated actions, where value have been 
            clipped to to be in range of values defined in the range parameter.
        '''
        acts = torch.zeros((1, self._action_size))
        #Evaluation mode for the local actor
        self.actor_local.eval()
        with torch.no_grad():
            acts[:] = self.actor_local(state)
        self.actor_local.train()
        if use_noise:
            acts += self._noise.sample()
        return torch.clip(acts, *range)

    def reset(self) -> None:
        '''Resets the noise used internally'''
        self._noise.reset()

    def save_model(self, path: Union[str, PathLike]) -> None:
        '''
        Saves the current state of the model to the disk. Can be used to create checkpoints, from
        which the training/evaluation of the model can be continued on.

        Parameters
        ----------
        path: str | PathLike
            Path to the file, where the model will be saved to.
        '''
        path = pathlib.Path(path) if not isinstance(path, pathlib.PurePath) else path

        #Create possible non-existent parent directories
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        #Save all the needed attributes
        state = {
            "actor_local_state": self._actor_local.state_dict(),
            "actor_target_state": self._actor_target.state_dict(),
            "critic_local_state": self._critic_local.state_dict(),
            "critic_target_state": self._critic_target.state_dict(),
            "actor_optimizer_state": self._actor_optim.state_dict(),
            "critic_optimizer_state": self._critic_optim.state_dict(),
            "noise": self._noise,
            "buffer": self._buffer,
            "action_size": self._action_size,
            "state_size": self._state_size,
            "gamma": self._gamma,
            "tau": self._tau
        }

        try:
            torch.save(state, path)
        except Exception as e:
            backup_path = timestamp_path(_BACKUP_PATH)
            warnings.warn(f"Error while trying to save the model to location {path}. Trying backup location {backup_path}. Error-msg: {e}")
            torch.save(state, backup_path)


    def load_model(self, path: Union[str, PathLike], eval: bool = False) -> None:
        '''
        Loads a previous state of the model from the disk, and moves it to the currently
        used device.

        Parameters
        ----------
        path: Union[str, PathLike]
            Path to the file, where the model is saved to.
        eval: bool, Optional
            Controls if the model is set to evaluation or training mode after it has been loaded.
            Default, False (i.e. the model is in training mode)

        Raises
        ------
        FileNotFoundError
            If the given path cannot be found
        KeyError
            If the loaded model isn't a correct type (i.e. it doens't represent this model)
        '''
        path = pathlib.Path(path) if not isinstance(path, pathlib.PurePath) else path

        if not path.exists():
            raise FileNotFoundError(f"Cannot find file at {path}")
        
        state = torch.load(path)

        self._actor_local.load_state_dict(state["actor_local_state"])
        self._actor_local.to(self._device)
        self._actor_target.load_state_dict(state["actor_target_state"])
        self._actor_target.to(self._device)

        self._critic_local.load_state_dict(state["critic_local_state"])
        self._critic_local.to(self._device)
        self._critic_target.load_state_dict(state["critic_target_state"])
        self._critic_target.to(self._device)


        self._actor_optim.load_state_dict(state["actor_optimizer_state"])
        self._critic_optim.load_state_dict(state["critic_optimizer_state"])

        self._noise = state["noise"]
        self._buffer = state["buffer"]

        self._action_size = state["action_size"]
        self._state_size = state["state_size"]

        self._gamma = state["gamma"]
        self._tau = state["tau"]

        if eval:
            self._actor_local.eval()
            self._actor_target.eval()
            self._critic_local.eval()
            self._critic_target.eval()
        else:
            self._actor_local.train()
            self._actor_target.train()
            self._critic_local.train()
            self._critic_target.rain()

    def learn(self, experiences: Tuple[torch.Tensor,torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        '''
        Updates the policy, and value parameters, using the given experiences. The used formula
        Qtargets = r + y * Q-value,
        where actor-target-network produces actions from the states, and critic-target network
        calculates the Q-values from (state, action) pairs.    

        Parameters
        ----------
        experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of tensors, that contain states, actions, rewards, next states, and information about if the
            task was completed at that point.
        '''
        states, actions, rewards, next_states, dones = experiences
        
        # --------------- Update the critic networks ------------------
        next_actions = self._actor_target(next_states)
        next_q_val_targets = self._critic_target(next_states, next_actions)

        #         
        q_targets = rewards + (self._gamma * next_q_val_targets * (1 - dones)) 
        q_expected = self._critic_local(states, actions)
        critic_loss = F.mse_loss(q_targets, q_expected)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        # -------------------- Update the actor ------------------------
        actions_pred = self._actor_local(states)
        actor_loss = -self._critic_local(states, actions_pred).mean()

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        # ---------------- Update the target networks ------------------
        self.soft_update(self._critic_local, self._critic_target, self._tau)
        self.soft_update(self._actor_local, self._actor_target, self._tau)

    def soft_update(self, local_network: nn.Module, target_network: nn.Module, tau: float) -> None:
        '''
        Soft update for the networks parameters. Ensures that the target networks values 
        always change atleast a little.
        target = tau * local + (1 - tau) * target 

        Parameters
        ----------
        local_network: nn.Module
            A model, where the values will be copied from.
        target_network: nn.Module
            A model. where the values will be update to.
        tau: float
            A control parameter, that is used as interpolation value.

        '''
        for t_param, l_param in zip(target_network.parameters(), local_network.parameters()):
            t_param.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    
