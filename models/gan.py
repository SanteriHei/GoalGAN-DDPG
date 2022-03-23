import torch
import torch.nn as nn
from torch import optim
import numpy as np

import matplotlib.pyplot as plt

import pathlib
from dataclasses import dataclass



import numpy.typing as npt
from os import PathLike
from typing import Union, Optional

import utils

_BACKUP_PATH = pathlib.Path("checkpoints/backups/lsgan.tar")

@dataclass
class GANConfig:
    '''
    Defines a configuration object for Generator/Disciminator in a GAN network

    Attributes
    ----------
    input_size: int, Optional
        The size of the inputs to the network. Default 4.
    hidden_size: int, Optional
        The size of the hidden layers. Default 256.
    output_size: int, Optional
        The size of the networks output. Default 2.
    layer_count: int, Optional
        The amount of linear layers used in the network. Default 2.
    opt_lr: float, Optional
        The learning rate of the optimizer used. Default 0.01
    opt_alpha: float, Optional
        The smoothing constant of the optimizer used. Default 0.99
    opt_momentum: float, Optional
        The momentum factor of the optimizer used. Default 1e-3. 
    '''
    input_size: int = 4
    hidden_size: int = 256
    output_size: int = 2
    layer_count: int = 2
    opt_lr: float = 0.01
    opt_alpha: float = 0.99
    opt_momentum: float = 1e-3




class LSGAN:
    '''
    Creates a LSGAN network, consisting of Generator and Discriminator networks

    Attibutes
    ---------
    generator: nn.Module
        The used generator network
    discriminator: nn.Module
        The used discriminator network
    generator_optimizer: torch.optim.RMSprop
        The optimizer used for the generator
    discriminator_optimizer: torch.optim.RMSprop
        The optimizer used for the discriminator
    device: torch.Device
        The device where the generator and discriminator networks are located

    
    Parameters
    ----------
    generator_config: Config
        A configuration object defining the following sizes for the generator
            - input_size
            - size of the hidden layers
            - output size
        Defines also following parameters for the optimizer
            - learning rate
            - alpha
            - momentum
    discriminator_config: Config
        A configuration object defining the following sizes for the discriminator
            - Input size
            - size of the hidden layers
            - output size
        Defines also the following parameters for the optimizer
            - learning rate
            - alpha
            - momentum
    device: str | torch.Device
        The device where the model will be moved to. Can be either a string presentation of the device, or
        a actual torch.device.
    a: int
        The label for fake data, (See LSGAN paper for more details)
    b: int
        The label for real data (See LSGAN paper for more details)
    c: int
        The value that generator want's D to believe for fake data (See LSGAN paper for more details) 
    '''

    def __init__(self, generator_config: GANConfig, discriminator_config: GANConfig, device: Union[str, torch.device], a: int = -1, b: int = 1, c: int = 0) -> None:      
        self._device = torch.device(device) if isinstance(device, str) else device
        
        #Create Generator and optimizer based on the defined config
        self._generator = Generator(
            generator_config.layer_count, generator_config.input_size, 
            generator_config.hidden_size, generator_config.output_size
        ).to(self._device)
        
        self._discriminator = Discriminator(
            discriminator_config.layer_count, discriminator_config.input_size, 
            discriminator_config.hidden_size, discriminator_config.output_size
        ).to(self._device)

        #Create optimizers using the set hyperparameters
        self._generator_optimizer = optim.RMSprop(
            self._generator.parameters(), lr=generator_config.opt_lr, 
            alpha=generator_config.opt_alpha, momentum=generator_config.opt_momentum
        )
        
        self._discriminator_optimizer = optim.RMSprop(
            self._discriminator.parameters(), lr=discriminator_config.opt_lr, 
            alpha=discriminator_config.opt_alpha, momentum=discriminator_config.opt_momentum
        )

        self._generator_input_size = generator_config.input_size        

        self._a = a
        self._b = b
        self._c = c

        #For logging to console and tensorboard
        self._logger = utils.get_logger(__name__)
        self._writer = utils.get_writer()
        self._discriminator_losses = []
        self._generator_losses = []


        #Apply Xavier uniform weight initialization
        self._generator.apply(init_weights)
        self._discriminator.apply(init_weights)

    def close(self) -> None:
        ''' 
        Closes the tensorboard writer, and saves the loss data. 
        Call only if all training is done.
        '''
        generator_losses = np.array(self._generator_losses)
        discriminator_losses = np.array(self._discriminator_losses)
        
        np.save("checkpoints/generator_loss.npy", generator_losses)
        np.save("checkpoints/discriminator_loss.npy", discriminator_losses)
        self._writer.close()

    def reset_weights(self) -> None:
        '''
        Resets the weights of the generator and discriminator
        '''
        self._generator.apply(init_weights)
        self._discriminator.apply(init_weights)

    @property
    def generator_input_size(self) -> int:
        '''
        Returns the input dimension of the generator
        
        Returns
        ------
        int:
            The input dimension of the generator
        '''
        return self._generator_input_size

    def generate_goals(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Generates goals from a noise vector by 
        applying the forward pass to the generator.
        NOTE: the network should be set to evaluation mode before calling this

        Parameters
        ----------
        z: torch.Tensor
            The input to the generator. Usually noise vector.
            see generator_input_size to create correct size noise.

        Returns
        -------
        torch.tensor
            The output of the generator network
        '''
        if z.device != self._device:
            z = z.to(self._device)
        return self._generator.forward(z)

    def save_model(self, path: Union[str, PathLike]) -> None:
        '''
        Saves the model's current state to the disk. Allows to continue to use the model from this
        state by calling model.load_model(path)

        Parameters
        ----------
        path: str | PathLike
            Path to location where the model will be saved to. Convention is to use .tar file-extension.
            Needed parent directories that don't exist will be created.
        '''
        path = pathlib.Path(path) if not isinstance(path, pathlib.PurePath) else path

        assert path.suffix == ".tar", f"The convention is to store the models using the .tar suffix, but got {path.suffix}"
        
        #Create all subdirectories on the way
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        #Create the state that contains all necessary parameters
        state = {
            "generator_state": self._generator.state_dict(),
            "discriminator_state": self._discriminator.state_dict(),
            "g_optimizer_state": self._generator_optimizer.state_dict(),
            "d_optimizer_state": self._discriminator_optimizer.state_dict(),
            "generator_input_size": self._generator_input_size,
            "a": self._a,
            "b": self._b,
            "c": self._c
        }

        try: 
            torch.save(state, path)
        except Exception as e: #as torch doesn't speficy what exceptions can be thrown here, use catch all
            backup_path = utils.timestamp_path(_BACKUP_PATH)
            self._logger.warning(f"Error while trying to save the model to location {path}. Trying backup location {backup_path}. Error-msg: {e}")
            torch.save(state, backup_path)

    def load_model(self, path: Union[str, PathLike], eval: bool = False) -> None:
        '''
        Loads a previous state of the model from the disk, and moves it to the 
        currently used device.

        Parameters
        ----------
        path: str | PathLike
            Path to file where the previous checkpoint of the model is located.
        eval: bool, Optional
            Controls if the model is set to evaluation or training mode after the loading.
            Default false (i.e. the model is in training mode)
        
        Raises
        ------
        FileNotFoundError
            If the given path cannot be found.
        KeyError
            If the loaded model isn't a correct type (i.e. it doesn't represent this model)
        '''
        path = pathlib.Path(path) if not isinstance(path, pathlib.PurePath) else path

        if not path.exists():
            self._logger.error(f"Cannot found file at {path}")
            raise FileNotFoundError(f"Cannot found file at {path}")
        state = torch.load(path)

        #Update the models and move them to the current device
        self._generator.load_state_dict(state["generator_state"])
        self._generator.to(self._device)  

        self._discriminator.load_state_dict(state["discriminator_state"])
        self._discriminator.to(self._device)

        self._generator_optimizer.load_state_dict(state["g_optimizer_state"])
        self._discriminator_optimizer.load_state_dict(state["d_optimizer_state"])
        
        self._generator_input_size= state["generator_input_size"]

        self._a = state["a"]
        self._b = state["b"]
        self._c = state["c"]

        if eval:
            self._generator.eval()
            self._discriminator.eval()
        else:
            self._generator.train()
            self._discriminator.train()
        

    def _discriminator_loss(self, y: torch.Tensor, z: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        '''
        Defines the loss function for the Discriminator. See equation (5) from Florensa et al. (2018)

        Parameters
        ----------
        y: torch.Tensor
            Binary labels, allows for training of 'negative' samples when y = 0
        z: torch.Tensor
            Noise vector used to generate goals.
        goals: torch.Tensor
            The current goals
        
        Returns
        -------
        torch.Tensor
            The value of the loss function that should be minimized
        '''
        #Short hands to make the formula easier to read
        g = self._generator
        d = self._discriminator
        return y * (d.forward(goals) - self._b)**2 + (1-y) * (d.forward(goals) - self._a)**2 + (d.forward(g.forward(z)) - self._a)**2

    def _generator_loss(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Defines the loss function for the Generator. See equation (5) from Florensa et al. (2018)
        
        Parameters
        ----------
        z: torch.Tensor
            Noise vector, which is used to generate goals.
        
        Returns
        -------
        torch.Tensor
            The value of the loss function that should be minimized.
        '''
        #Short hands to make the formula more readable
        g = self._generator
        d = self._discriminator
        return (d.forward(g.forward(z)) - self._c)**2

    def _log_losses(self, gloss: npt.NDArray, dloss: npt.NDArray, global_step: Optional[int]) -> None:
        '''
        Logs the losses from the generator and discriminator to the tensorboard

        Parameters
        ----------
        gloss: npt.NDArray
            The generators loss values from the last training run
        dloss: npt.NDArray
            The discriminators loss values from the last training run
        global_step: int
            The current global iteration of the training loop.

        '''

        self._logger.debug(f"ITERATION {global_step}: Logging GAN's losses")
        x = np.arange(len(gloss))
        
        fig_gloss = utils.line_plot(
            x, gloss, close_fig=False, title=f"Generator loss {global_step}", 
            ylabel="Loss", xlabel="Iteration", figsize=(25, 10)
        )

        fig_dloss = utils.line_plot(
            x, dloss, close_fig=False, title=f"Discriminator loss {global_step}", 
            ylabel="Loss", xlabel="Iteration", figsize=(25, 10)
        )

        self._writer.add_figure("loss/generator", fig_gloss, global_step=global_step)
        plt.close(fig_gloss)
        
        self._writer.add_figure("loss/discriminator", fig_dloss, global_step=global_step)
        plt.close(fig_dloss)



    def train(self, goals: torch.Tensor, labels: npt.NDArray[np.int32], n_iter: int = 200, global_step: Optional[int] = None) -> None:
        '''
        Trains the gan network for given amount of iterations

        Parameters
        ----------
        goals: torch.Tensor
            The current goals
        labels: npt.NDArray[np.int32]
            The goals for the goals. 1 means that the goals was in GOID, 0 that it wasn't, i.e. 
            the goal was either too easy or too difficult for the agent.
        n_iter: int, Optional
            The amount of iterations that the GAN is trained for. Default 10.
        '''
        y = torch.from_numpy(labels).unsqueeze(dim=-1).float().to(self._device) #Add 'dummy' dimension
        
        goals = goals.float()
        if goals.device != self._device:
            goals = goals.to(self._device)

        self._logger.info(f"Training GAN for {n_iter} iterations")
        dloss = np.zeros((n_iter,))
        gloss = np.zeros((n_iter,))

        for i in range(n_iter):
            
            #--------Update the Discriminator-----------

            #The size of the noise must be same as the input size of the generator
            zs = torch.randn(len(labels), self._generator_input_size).to(self._device)          
            self._discriminator.zero_grad()
            discriminator_loss = torch.mean(self._discriminator_loss(y, zs, goals))
            discriminator_loss.backward()
            self._discriminator_optimizer.step()
            
            #---------Update the Generator-------------
   
            zs = torch.randn(len(labels), self._generator_input_size).to(self._device)
            self._generator.zero_grad()

            generator_loss = torch.mean(self._generator_loss(zs))
            generator_loss.backward()
            self._generator_optimizer.step()
            
            #--------Save loss data for logging purposes----------------

            self._discriminator_losses.append(discriminator_loss.item())
            dloss[i] = discriminator_loss.item() 

            self._generator_losses.append(generator_loss.item())
            gloss[i] = generator_loss.item()
        
        #Update the tensorboard
        self._log_losses(gloss, dloss, global_step)

 

class Generator(nn.Module):
    '''
    Generator network for goal GAN

    Attributes
    ----------
    layers: nn.ModuleList
        List of layers that define the generator
    
    Parameters
    ----------
    n_layers: int
        The amount of linear layer added to the network. Must be positive
    input_size: int
        The size of the data given to this network
    hidden_size: int
        The size of the hidden layers in this network.
    output_size: int
        The size of the output produced by this network.
    
    '''
    def __init__(self, n_layers, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        assert n_layers > 0, f"There must be atleast 1 layer,({n_layers} was given)"
        self._layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        ])
        for i in range(n_layers):
            if i < n_layers - 1:
                self._layers.append(nn.Linear(hidden_size, hidden_size))
                self._layers.append(nn.ReLU())
            else:
                self._layers.append(nn.Linear(hidden_size, output_size))
                self._layers.append(nn.Tanh())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies a forward pass to the generator
        Parameters
        ----------
        x: torch.Tensor
            The input data for the generator

        Returns
        -------
        x: torch.Tensor
            The data after it has been applied to the generator network.
        '''
        for layer in self._layers:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    '''
    Discriminator network for goal GAN

    Attributes
    ----------
    layers: nn.ModuleList
        List of layers that define the discriminator

    Parameters
    ----------
    n_layers: int
        The amount of linear layers used in the Discriminator.
    input_size: int
        The size of the input data given to the discriminator.
    hidden_size: int
        The size of the hidden layers of the discriminator.
    output_size: int
        The size of the output produced by the discriminator. 

    '''
    def __init__(self, n_layers, input_size, hidden_size, output_size) -> None:
        super(Discriminator, self).__init__()
        assert n_layers > 0, f"There must be atleast 1 layer ({n_layers} was given)"

        self._layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU()
        ])

        for i in range(n_layers):
            if i < n_layers - 1:
                self._layers.append(nn.Linear(hidden_size, hidden_size))
                self._layers.append(nn.LeakyReLU())
            else:
                self._layers.append(nn.Linear(hidden_size, output_size))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies a forward pass to the data
        
        Parameters
        ----------
        x: torch.Tensor
            The input data for the network. Must have dimensions comparable to the 'input_size' used when creating this layer.

        Returns
        -------
        x: torch.Tensor
            The resulting data after the discriminator network has been applied to it. 

        '''
        for layer in self._layers:
            x = layer(x)
        return x



#SO: https://stackoverflow.com/a/49433937
def init_weights(m: nn.Module):
    '''
    Initializes the weights of a module with Xavier uniform initialization

    Parameters
    ----------
    m: nn.Module
        The torch module to apply the weight initialization
    
    Returns
    ------
    None
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)