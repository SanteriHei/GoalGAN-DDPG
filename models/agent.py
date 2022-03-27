
from models.rl import Actor, Critic, OUNoise, MemoryBuffer, Experience

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple, Union
from os import PathLike
import numpy.typing as npt

import pathlib
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
    action_size: int
        Defines the size of each action. The action MUST have 1-dimensional size, 
        i.e. the tuple defining it's size has only 1 element.
    state_size: int 
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
    action_size: int = _NOT_SET
    state_size: int = _NOT_SET
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    weight_decay: float = 0.0
    tau: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = int(1e5)
    batch_size: int = 128

    def __str__(self) -> str:
        '''Returns a simple table presentation of the config values'''
        header = f"{'action-size':^11s}|{'state-size':^10s}|{'actor-lr':^8s}|{'critic-lr':^9s}|{'weight-decay':^12s}|{'tau':^7s}|{'gamma':^9s}|{'buffer-size':^11s}|{'batch-size':^10s}"
        delim = f"{11*'-'}|{10*'-'}|{8*'-'}|{9*'-'}|{12*'-'}|{7*'-'}|{9*'-'}|{11*'-'}|{10*'-'}"
        values = f"{self.action_size:^11d}|{self.state_size:^10d}|{self.actor_lr:^8.4f}|{self.critic_lr:^9.4f}|{self.weight_decay:^12.6f}|{self.tau:^7.5f}|{self.gamma:^9.4f}|{self.buffer_size:^11d}|{self.batch_size:^10d}"
        return f"{header}\n{delim}\n{values}"

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

        self._logger = utils.get_logger(__name__)
        self._writer = utils.get_writer()
        self._critic_losses = []
        self._actor_losses  = []


    def close(self) -> None:
        ''' Closes the tensorboard writer. Call only if training is done'''
        self._writer.close()

    def _check_dim_and_unwrap(self, value: Union[int, Tuple[int]]) -> int:
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

    def step(self, state: npt.NDArray, action: npt.NDArray, reward: float, next_state: npt.NDArray, done: bool) -> None:
        '''
        Takes a step with the action, and updates the replay buffer. If enough data is 
        saved, the agent will be trained

        Parameters
        ----------
        state: npt.NDArray
            The state before the action.
        action: npt.NDArray
            The action that was taken
        reward: float
            The received reward.
        next_state: npt.NDArray
            The state that was achieved by taking the action.
        done: bool 
            Was the task completed.
        '''
        exp = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self._buffer.append(exp)

        if len(self._buffer) > self._buffer.batch_size:
            experiences = self._buffer.sample()
            self.learn(experiences)


    def act(self, state: npt.NDArray, use_noise: bool = True, range: Tuple[float, float] = (0.0, 1.0)) -> npt.NDArray:
        '''
        Uses the current policy to create actions for the given state

        Parameters
        ----------
        state: npt.NDArray
            The current state.
        use_noise: bool, Optional
            Controls if additional noise is added to the 
            generated actions. Default True.
        range: Tuple[int, int], Optional
            Defines the range, where the values should be clipped to.
        Returns
        -------
        npt.NDArray
            The generated actions, where value have been 
            clipped to to be in range of values defined in the range parameter.
        '''

        state = torch.from_numpy(state).float().to(self._device)
        acts = torch.zeros((1, self._action_size)).float().to(self._device)

        #Evaluation mode for the local actor
        self._actor_local.eval()
        with torch.no_grad():
            acts[:] = self._actor_local.forward(state)
        
        self._actor_local.train()
        
        if use_noise:
            acts += torch.from_numpy(self._noise.sample()).to(self._device)
        
        #Convert action back to numpy array
        cpu_action = acts.detach().cpu().numpy()
        return np.clip(cpu_action, *range).squeeze()


    def log_losses_and_reset(self, global_step: Optional[int] = None) -> None:
        '''
        Logs the losses to the tensorboard by creating figures, and resets them.

        Parameters
        ----------
        global_step: Optional[int]
            The global training step.
        '''

        self._logger.debug(f"ITERATION {global_step} Logging losses ")

        x = np.arange(len(self._actor_losses))
        actor_loss = np.array(self._actor_losses)
        critic_loss = np.array(self._critic_losses)

        fig_actor_loss = utils.line_plot(
            x, actor_loss, title=f"Actor loss {global_step}", 
            xlabel="iterations", ylabel="loss", figsize=(20, 10), grid=True
        )

        fig_critic_loss = utils.line_plot(
            x, critic_loss, title=f"Critic loss {global_step}", 
            xlabel="iterations", ylabel="loss", figsize=(20, 10), grid=True
        )

        self._writer.add_figure("loss/actor", fig_actor_loss, global_step=global_step)
        self._writer.add_figure("loss/critic", fig_critic_loss, global_step=global_step)

        self._actor_losses = []
        self._critic_losses = []

        plt.close(fig_actor_loss)
        plt.close(fig_critic_loss)


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
            backup_path = utils.timestamp_path(_BACKUP_PATH)
            self._logger.warning(f"Error while trying to save the model to location {path}. Trying backup location {backup_path}. Error-msg: {e}")
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
            self._critic_target.train()

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
        next_actions = self._actor_target(next_states) #(batch_size, action_size)
        next_q_val_targets = self._critic_target(next_states, next_actions) #(batch_size, 1)

        q_targets = rewards.unsqueeze(-1) + (self._gamma * next_q_val_targets * (1 - dones.unsqueeze(-1)))  #(batch_size, 1)
        q_expected = self._critic_local(states, actions) 
        
        critic_loss = F.mse_loss(q_targets, q_expected)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()


        # -------------------- Update the actor ------------------------
        actions_pred = self._actor_local(states)
        actor_loss = -torch.mean(self._critic_local(states, actions_pred))

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        # ---------------- Update the target networks ------------------
        self.soft_update(self._critic_local, self._critic_target, self._tau)
        self.soft_update(self._actor_local, self._actor_target, self._tau)


        #----------------- Store losses ---------------------------
        self._actor_losses.append(actor_loss.item())
        self._critic_losses.append(critic_loss.item())



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
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    
