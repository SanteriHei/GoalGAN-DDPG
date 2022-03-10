from time import time
import torch
import numpy as np
import pathlib

from models.gan import LSGAN
from models.agent import DDPGAgent
from environment.mujoco_env import MazeEnv
import utils

from typing import Sequence, Optional, Union
from os import PathLike
import numpy.typing as npt

_MAX_T_STEPS: int = 500
_MAX_EPISODES: int = 10

_logger = utils.get_logger(__name__)


def _eval_and_update_policy(
    agent: DDPGAgent, env: MazeEnv, goals: Sequence[npt.NDArray], 
    episode_count: int, timestep_count: int
    ) -> npt.NDArray[np.float32]:
    '''
    Evaluates the agents current policy for the current goals, and adds the experience to the 
    agents relay. If enough experiences are collected, the agent will update itself.

    Parameters
    ----------
    agent
        The policy-agent that will be evaluated
    env
        The enviroment, where the policy agent will try 
        to reach the goals
    goals: Sequence[npt.NDArray]
        The goals for the agent to reach.
    episode_count: int
        The amount of episodes evaluated
    timestep_count: int
        The maximum amount of timesteps during each episode 

    Returns
    -------
    np.NDArray[np.float32]
        The amount of times a given goal was reached, normalized to be in range (0, 1).

    '''
    _logger.info("Evaluation and updating policy")
    env.goals = goals #Set the current goals for the agent.
    for i in range(episode_count):
        state = env.reset()
        for j in range(timestep_count):
            action = agent.act(torch.from_numpy(state)) #Create the action based on the current state.
            cpu_action = action.detach().cpu().numpy().squeeze()
            next_state, reward, done = env.step(cpu_action)

            agent.step(state, cpu_action, reward, next_state, done)
            #If any of the goals was achieved during the episode, we stop and start a new episode
            if done:
                _logger.info(f"(Episode {i}): Goal found in {j} timesteps")
                break 

    # Check how many times each goal was reached during the training, 
    # and normalize the values between (0,1) for goal labeling
    _logger.info("Evaluation & updating done")
    return env.achieved_goals_counts / _MAX_EPISODES 


def _update_replay(current_goals: torch.Tensor, old_goals: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    '''
    Implementation of a 'buffer' of old goals, so that the model doesn't "forget" 
    how to achieve the old goals.

    Parameters
    ----------
    current_goals: torch.Tensor
        The goals generated during current iteration
    old_goals: torch.Tensor
        The goals from older iterations
    eps: float
        The precision used to check if given points are close to each other

    Returns
    -------
    torch.Tensor:
        A new set of goals, that will act as old goals for next iteration
    '''
    for g in current_goals:
        #TODO: better mechanism for this loop thing
        is_close = min((torch.dist(g, old_g) for old_g in old_goals)) < eps
    
        #If the goal isn't close to the old, add it to the list of old goals
        if not is_close:
            old_goals = torch.cat((g.unsqueeze(-1).T, old_goals))
    return old_goals


def _initialize_gan(
    gan: LSGAN, agent: DDPGAgent, env: MazeEnv, iter_count: int, 
    goal_count: int, episode_count: int, timestep_count: int
    ) -> torch.Tensor:
    '''
    Initializes the Goal-GAN and produces somewhat easy goals for the agent to use at the start. 
    See Appendix A.2 from Florenso et al. 2018 for more detailed explanation of the problem solved here.

    Parameters
    ----------
    gan: LSGAN
        The GAN network used in the goal gan
    agent: DDPGAgent
        The agent used explore the environment.
    env: MazeEnv
        The environment, where the goals are produced to.
    iter_count: int
        The amount of iterations is used in the pre-training phase.
    goal_count: int
        The amount of goals to produce during each iteration.
    episode_count: int
        The amount of episodes evaluated on each set of goals.
    timestep_count: int
        The maximum amount of timesteps during each episode.
    Returns
    -------
    torch.Tensor:
        The goals that should be suitable for the agent.
    '''

    for i in range(iter_count):
        if (i+1)%10 == 0:
            _logger.info(f"Pretrain: iteration: {i}")
        starting_pos = torch.from_numpy(env.agent_pos)
        goals = torch.clamp(starting_pos + 0.1*torch.randn(goal_count, env.goal_size), *env.limits)
        
        #Train the agent with the randomly generated goals
        returns = _eval_and_update_policy(agent, env, goals.detach().cpu().numpy(), episode_count, timestep_count)
        #Label the goals and train the GAN
        labels = utils.label_goals(returns)
        gan.train(goals, labels)

    #After the Policy is trained with the random goals, we choose random, easy goals from the close neighborhood of the agent.
    env.reset()
    goals = torch.from_numpy(env.agent_pos).unsqueeze(0) + 0.1*torch.randn(goal_count, env.goal_size)
    return torch.clamp(goals, *env.limits)


def train(
    gan: LSGAN, agent: DDPGAgent, env: MazeEnv, pretrain_iter_count: int, iter_count: int,
    goal_count: int, episode_count: int, timestep_count: int,
    save_after: Optional[int] = None, gan_base_path: Optional[Union[str, PathLike]] = None, 
    ddpg_base_path: Optional[Union[str, PathLike]] = None
) -> None:
    '''
    Trains the Goal gan by using the originally proposed training algorithm from the paper
    z = sample_noise
    goals = Generator(z) union old samples (concatenate together)
    policy = update_policy(goals, old_policy)
    returns = evaluate_policy(goals, policy)
    labels = label_goals(returns)
    train_gan(goals, labels)

    Parameters
    ----------
    gan: LSGAN
        The GAN network that is used to generate the goals
    agent: DDPGAgent
        The agent that explores the state space
    env: MazeEnv
        The enviroment, where the agent explores the space
    pretrain_iter_count: int
        The amount of pretraining iterations. During these iterations, the policy is trained 
        using only the randomly generated data.
    iter_count: int
        The amount of actual training iterations used.
    goal_count: int
        The amount of goals generated per each iteration
    episode_count: int
        The amount of episodes evaluated on each set of goals
    timestep_count: int
        The maximum amount of timesteps on each episode 
    save_after: Optional[int]
        Define how often the models are saved (i.e. after how many iterations). If not 
        specified, the models won't be saved.
    gan_base_path: Optional[Union[str, PathLike]]
        The base path, where the gan model's states will be saved during the training. The actual filenames
        will contain iteration numbers, that will specify the time of the training process, when the model was saved.
    ddpg_base_path: Optional[Union[str, PathLike]]
        The base path, where the DDPG agent's model states will be saved during the training. The actual 
        filenames will contain iteration counts, that will specify the time of the training process when the model was saved.
    '''

    if save_after is not None and (gan_base_path is None and ddpg_base_path is None):
        raise ValueError("If save_after is specified, then atleast one of the following must be specified: gan_base_path, ddpg_base_path")
    
    ddpg_base_path = pathlib.Path(ddpg_base_path) if not isinstance(ddpg_base_path, pathlib.PurePath) else ddpg_base_path
    gan_base_path = pathlib.Path(gan_base_path) if not isinstance(gan_base_path, pathlib.PurePath) else gan_base_path 

    #50% gan generated goals and 50% of random goals
    random_goal_count = goal_count // 2
    gan_goal_count = goal_count // 2
    
    #Save the mean value of the returns to plot a "coverage" graph
    avg_coverages = np.zeros((iter_count, ), dtype=np.float64)

    _logger.info(f"Starting pre-training")
    old_goals = _initialize_gan(gan, agent, env, pretrain_iter_count, goal_count, episode_count, timestep_count)
    _logger.info("Ending pre-training")
        
    _logger.info("Starting training")    
    for i in range(iter_count):
        if (i + 1)%10 == 0:
            utils.display_agent_and_goals(env.agent_pos, goals.detach().cpu().numpy(), env.limits, filepath=f"images/img_{i}.png", pos_label="Agent position", title=f"Iteration {i}", goal_label="goals")
            _logger.info(f"Training iteration {i}, created figure")

        #Sample noise
        z = torch.randn((gan_goal_count, gan.generator_input_size)) 

        #Create goals from the noize
        gan_goals = gan.generate_goals(z).detach()

        #Ensure that all goals are in same device
        rand_goals = torch.Tensor(random_goal_count, env.goal_size).uniform_(-1, 1).to(gan_goals.device)
        old_goals = old_goals.to(gan_goals.device)
        
        #Use 50% of random goals, and 50% of generated goals (See Appendix B.4 from Florensa et al. 2018)
        goals = torch.cat([gan_goals, utils.sample_tensor(old_goals, random_goal_count), rand_goals])

        #Evaluate & update the agent.
        returns = _eval_and_update_policy(agent, env, goals.detach().cpu().numpy(), episode_count, timestep_count)
        avg_coverages[i] = np.mean(returns)

        #Label the goals to be either in the GOID or not. Range 0.1 <= x <= 0.9 is used as in the original paper.
        labels = utils.label_goals(returns)

        if np.all(labels == 0):
            _logger.warning(f"All labels 0 during training iteration {i+1}")

        #Train the Goal GAN with the goals and their labels.
        gan.train(goals, labels)

        #Update the old goals to ensure that "catastrophic forgetting"
        old_goals = _update_replay(gan_goals, old_goals)
        
        #Save the models
        if (i + 1) % save_after == 0:
            ddpg_path = utils.add_to_path(ddpg_base_path, "iter_{i}")
            agent.save_model(ddpg_path)
            _logger.info(f"Iteration {i}: Saved DDPG to {ddpg_path}")

            gan_path = utils.add_to_path(gan_base_path, "iter_{i}")
            gan.save_model(gan_path)
            _logger.info(f"Iteration {i}: Saved gan to {gan_path}")
    
    x = np.arange(0, iter_count)
    utils.line_plot_1d(x, avg_coverages, "figures/training_coverage.png", title="Training coverage", xlabel="Iterations", ylabel="Avg coverage")
    _logger.info("Training done!")


