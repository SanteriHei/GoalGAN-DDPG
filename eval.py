from models.agent import DDPGAgent
from models.gan import LSGAN
from environment.mujoco_env import MazeEnv
import utils
import torch
import numpy as np


from typing import Union, Optional
from os import PathLike
import numpy.typing as npt

_logger = utils.get_logger(__name__)


def _learn_policy(agent: DDPGAgent, env: MazeEnv, episode_count: int, max_timestep_count: int) -> npt.NDArray:
    '''
    Evaluates and learns the policy using the currently set goals.

    Parameters
    ----------
    agent: DDDPAgent
        The agent used to learn the policy
    env: MazeEnv
        The environment, where the goals are located.
    episode_count: int
        The amount of episodes for this set of goals.
    max_timestep_count: int
        The maximum amount of timesteps allowed per episode.
    
    Returns
    -------
    npt.NDArray
        Returns the amount of times each goal was reached, where the 
        counts have been normalized between [0, 1].
    '''
    _logger.info(f"Learning and evaluating policy for {episode_count} episodes")

    for i in range(episode_count):
        _logger.info(f"Episode {i} starting...")
        state = env.reset()
        for j in range(max_timestep_count):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)

            if done:
                _logger.info(f"(Episode {i}): Goal reached in {j} timesteps")
                break
    _logger.info("Evaluation and updating done")
    return env.achieved_goals_counts/episode_count

def eval_gan(
        gan: LSGAN, agent: DDPGAgent, env: MazeEnv, max_iter: int, 
        episode_count: int, max_timestep_count: int, goal_count: int, 
        figure_save_path: Optional[Union[str, PathLike]] = None, data_save_path: Optional[Union[str, PathLike]] = None
    ) -> None:
    '''
    Evaluates the performance of the Goal GAN
    
    Parameters
    ----------
    gan: LSGAN
        The GAN network to evaluate
    agent: DDPGAgent
        The agent to train using the Goal GAN
    env: MazeEnv
        The environment, where the goals will be generated to
    max_iter: int
        The amount of iterations in the outer loop
    episode_count: int
        The amount of episodes for each set of goals
    max_timestep_count: int
        The maximum amount of timesteps allowed per episode.
    goal_count: int 
        The amount of goals generated per set
    figure_save_path: Optional[str | PathLike]
        Path to file, where the figure of the performance will be saved to.
        If no path is specified, the image will not be generated. Default None.
    data_save_path: Optional[str | PathLike]
        Path to file, where the generated data will be saved to.
        If not path is specified, the data will not be saved. Default None.
    '''

    avg_coverages = np.zeros((max_iter,), dtype=np.float64)

    _logger.info(f"Starting evaluating GAN. Doing {max_iter} iterations")
    for i in range(max_iter):
        _logger.info(f"Iteration {i}")
        #Create the noise vector
        z = torch.randn((goal_count, gan.generator_input_size))
        goals = gan.generate_goals(z)
        # Set the goals for the environment
        env.goals = goals
        achieved_goals = _learn_policy(agent, env, episode_count, max_timestep_count)
        avg_coverages[i] = np.mean(achieved_goals)

    if data_save_path is not None:  
        arr_path = utils.timestamp_path(data_save_path)
        np.save(arr_path, avg_coverages)
        _logger.info(f"Data saved to {arr_path}")

    if figure_save_path is not None:
        fig_path = utils.timestamp_path(figure_save_path)
        x = np.arange(0, max_iter, 1)
        utils.line_plot(x, avg_coverages, fig_path)
        _logger.info(f"Figure saved to {fig_path}")
    _logger.info("Evaluation done")

