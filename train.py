import torch
import numpy as np
import warnings
import pathlib

from models.gan import LSGAN
from models.agent import DDPGAgent
from environment.mujoco_env import MazeEnv
import utils

from typing import Sequence, Optional, Union
from os import PathLike
import numpy.typing as npt

MAX_T_STEPS: int = 500
MAX_EPISODES: int = 10


def _eval_policy(agent, env, goals: Sequence[npt.NDArray]) -> npt.NDArray[np.float32]:
    '''
    Evaluates the agents current policy for the current goals

    Parameters
    ----------
    agent
        The policy-agent that will be evaluated
    env
        The enviroment, where the policy agent will try 
        to reach the goals
    goals: Sequence[np.ndarray]
        The goals for the agent to reach.
    
    Returns
    -------
    np.ndarray
        The amount of times a given goal was reached, normalized to be in range (0, 1).

    '''
    env.goals = goals #Set the current goals for the agent.
    for i in range(MAX_EPISODES):
        state = env.reset()
        for _ in range(MAX_T_STEPS):
            action = agent.act(state) #Create the action based on the current state.
            state, _, done = env.step(action)
            #If any of the goals was achieved during the episode, we stop and start a new episode
            if done:
                break 

    counts = env.achieved_goal_counts() #Check how many times each goal was reached during the training
    return counts/MAX_EPISODES #Normalize the values to be between (0,1) for goal labeling

    

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
            old_goals = torch.cat((g.view(-1), old_goals))
    return old_goals


def train(
    gan: LSGAN, agent: DDPGAgent, env: MazeEnv, pretrain_iter_count: int, iter_count: int,
    goal_count: int, save_after: Optional[int] = None, gan_base_path: Optional[Union[str, PathLike]] = None, 
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
    random_goals_count = goal_count // 2
    dim_env_goal_space = 0
        
    #Initially, the policy is trained using only random goals
    print(f"Starting pre-training")
    for i in range(pretrain_iter_count):
        if (i + 1)%10 == 0:
            print(f"[Iter: {i}]")

        starting_pos = torch.from_numpy(env.agent_pos)
        goals = torch.clamp(starting_pos + 0.1*torch.randn(goal_count, dim_env_goal_space), min=-1, max=1) #Clamp values between -1 and 1
        
        #Train the agent with the random goals, and "update the policy"
        agent.learn(goals)
        returns = _eval_policy(agent, env, goals)
        labels = utils.label_goals(returns)
        gan.train(goals, labels)

    old_goals = None

    print("Starting training")    
    for i in range(iter_count):
        if (i + 1)%10 == 0:
            print(f"[Iter: {i}]")

        #Sample noise
        z = torch.randn((goal_count, gan.generator_input_dim())) 

        #Create goals fromt the noize
        gan_goals = gan.generator_forward(z).detach()
        rand_goals = torch.Tensor(random_goals_count, dim_env_goal_space).uniform_(-1, 1)
        
        #Use 50% of random goals, and 50% of generated goals (See Appendix B.4 from Florensa et al. 2018)
        goals = torch.cat([gan_goals, utils.sample_tensor(old_goals, random_goals_count), rand_goals])

        #Update the current policy by training the agent.
        agent.learn(goals)

        #Evaluate the agent on the same goals?
        returns = _eval_policy(agent, env, goals)

        #Label the goals to be either in the GOID or not. Range 0.1 <= x <= 0.9 is used as in the original paper.
        labels = utils.label_goals(returns)

        if np.all(labels == 0):
            warnings.warn(f"All labels 0 during training iteration {i+1}")

        #Train the Goal GAN with the goals and their labels.
        gan.train(goals, labels)

        #Update the old goals to ensure that "catastrophic forgetting"
        old_goals = _update_replay(gan_goals, old_goals)
        
        #Save the models
        if (i + 1) % save_after == 0:
            print(f"[Iter: {i}]: Saving the models")
            ddpg_path = utils.add_to_path(ddpg_base_path, "iter_{i}")
            agent.save_model(ddpg_path)

            gan_path = utils.add_to_path(gan_base_path, "iter_{i}")
            gan.save_model(gan_path)



