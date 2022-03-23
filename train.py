import torch

import numpy as np
import pathlib

from models.gan import LSGAN
from models.agent import DDPGAgent
from environment.mujoco_env import MazeEnv
import utils

from typing import Dict, Sequence, Optional, Union, Tuple
from os import PathLike
import numpy.typing as npt


_logger = utils.get_logger(__name__)
_writer = utils.get_writer()


def _clean_up(avg_coverages: npt.NDArray, rewards: npt.NDArray, gan: LSGAN, agent: DDPGAgent) -> None:
    '''
    Convience function to save figures, and close all tensorboard writers when the training is done.

    Parameters
    ----------
    avg_coverages: npt.NDArray
        The average coverages of the training run
    rewards: npt.NDArray
        The average rewards from policy evaluation
    gan: LSGAN
        The GAN that was trained.
    agent: DDPGAgent
        The agent that was used in the training
    iter_count: int
        The amount of iterations used.

    '''
    
    np.save("checkpoints/avg_coverages.npy", avg_coverages)
    utils.line_plot(
        np.arange(avg_coverages.shape[0]), avg_coverages, close_fig=True, 
        filepath="images/training_coverage.png", title="Training coverage",
        xlabel="iterations", ylabel="avg coverage", figsize=(25, 10)
    )
    

    np.save("checkpoints/avg_rewards.npy", rewards)
    x = np.arange(rewards.shape[0])
    utils.line_plot(
        np.arange(rewards.shape[0]), rewards,  close_fig=True,
        filepath="images/avg_rewards.png", title="Average rewards",
        xlabel="Outer iteration", ylabel="Avg reward/episode", figsize=(25, 10) 
    )

    gan.close()
    agent.close()
    _writer.close()


def _update_or_eval_policy(
    agent: DDPGAgent, env: MazeEnv, policy_iter_count: int, episode_count: int, 
    timestep_count: int, eval_mode: bool = False, 
    global_rewards: Optional[npt.NDArray] = None, global_step: Optional[int] = None,
) -> Optional[npt.NDArray]:
    '''
    Update or evaluate the current policy. 

    Parameters
    ----------
    agent: DDPGAgent
        The agent whos policy is updated/evaluated
    env: MazeEnv
        The environment where the goals are located.
    policy_iter_count: int
        The amount of training/evaluation iterations done
    episode_count: int
        The amount of episodes executed during each policy iteration
    timestep_count: int
        The maximum amount of timesteps allowed during a episode.
    eval_mode: bool, optional
        Controls if the policy is updated or evaluated. If set to True,
        the policy is evaluated, otherwise it is updated. Default False.
    global_rewards: Optional[npt.NDArray]
        A pre-allocated array, where average rewards can be saved. 
    global_step: Optional[int]
        The current global training step. Default None.
    
    Returns
    -------
    Optional[npt.NDArray]
        The amount of times each goal was achieved during the evaluation.
        is returned only if eval_mode is set to True, otherwise None is returned.
    '''

    tag = "update" if not eval_mode else "eval"
    rewards = np.zeros((policy_iter_count*episode_count, ))
    env.eval = eval_mode
    
    for i in range(policy_iter_count):
        for ep in range(episode_count):
            state = env.reset()
            for ts in range(timestep_count):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                if not eval_mode:
                    agent.step(state, action, reward, next_state, done)

                rewards[i*ep] = reward
                if done:
                    _logger.info(f"(iter: {i}, Episode: {ep}): Goal found in {ts} timesteps")
                    break
    avg_reward = rewards.mean()
    global_rewards[global_step] = avg_reward

    _writer.add_scalar(f"reward/{tag}-avg-reward", avg_reward, global_step=global_step)
    if not eval_mode:
        agent.log_losses_and_reset(global_step)

    if eval_mode:
        return np.clip(env.achieved_goals_counts/(policy_iter_count*episode_count), 0,1)



def _observe_states(agent: DDPGAgent, env: MazeEnv, goals: Sequence[npt.NDArray], episode_count: int, timestep_count: int) -> npt.NDArray:
    '''
    Obserses the states that the agent visits with it's initial policy. The goals should be
    sampled uniformly from the whole goal space. See Appendix A.2 from Florensa et al. 2018 for more details

    Parameters
    ----------
    agent: DDPGAgent
        The agent that observes the new states
    env: MazeEnv
        The environment, where the goals are located in.
    goals: Sequence[npt.NDArray]
        The goals to use for this environment
    episode_count: int
        The amount of episodes used for the given goals
    timestep_count: int
        the maximum amount of episodes allowed per episode.
    
    Returns
    -------
    npt.NDArray
        The positions that agent visited while trying to reach the uniformly
        created goals

    '''
    _logger.info("Training on random goals")
    env.goals = goals
    visited_positions = np.zeros((episode_count*timestep_count, env.goal_size))
    for ep in range(episode_count):
        state = env.reset()
        for ts in range(timestep_count):
            action = agent.act(state)
            *_, done = env.step(action)
            
            visited_positions[ep*episode_count + ts, :] = env.agent_pos

            if done:
                _logger.info(f"Goal found at timestep {ts} during episode {ep}")
                break            
    return visited_positions


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
    
    #If there is no old goals, use all current goals as old goals for 
    #next iteration
    if old_goals is None:
        return current_goals


    for g in current_goals:
        #TODO: better mechanism for this loop thing
        is_close = min((torch.dist(g, old_g) for old_g in old_goals)) < eps
    
        #If the goal isn't close to the old, add it to the list of old goals
        if not is_close:
            old_goals = torch.cat((g.unsqueeze(-1).T, old_goals))
    return old_goals


def _initialize_gan(
    gan: LSGAN, agent: DDPGAgent, env: MazeEnv, gan_iter_count: int, 
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

    rng = np.random.default_rng()
    #Create the initial goals.
    goals = rng.uniform(*env.limits, size=(goal_count, env.goal_size))
    visited_positions = _observe_states(agent, env, goals, episode_count, timestep_count)
    labels = np.ones((max(visited_positions.shape), ))
    gan.train(torch.from_numpy(visited_positions), labels, gan_iter_count, global_step=1)



def _create_goals(gan: LSGAN, goal_count: int) -> torch.Tensor:
    '''
    Creates new set of goals for the Agent to learn.

    Parameters
    ----------
    gan: LSGAN
        The GAN network used to generate goals.
    goal_count: int 
        The amount of goals to generate with GAN.
    Returns
    -------
    torch.Tensor
        The goals generated by the GAN.
    '''

    #Sample noise
    z = torch.randn((goal_count, gan.generator_input_size)) 
    #Create goals from the noize
    return gan.generate_goals(z).detach()    
    

def train(
    gan: LSGAN, agent: DDPGAgent, env: MazeEnv, gan_iter_count: int, policy_iter_count: int, iter_count: int,
    goal_count: int, episode_count: int, timestep_count: int, rmin: float, rmax: float,
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
    gan_iter_count: int
        The amount of iterations that the GAN is trained for 
        during each outer iteration.
    policy_iter_count: int
        The amount of iterations that the policy is updated for
        during each outer iteration.
    iter_count: int
        The amount of actual training iterations used.
    goal_count: int
        The amount of goals generated per each iteration
    episode_count: int
        The amount of episodes evaluated on each set of goals
    timestep_count: int
        The maximum amount of timesteps on each episode 
    rmin: float
        The minimum return value considered to be feasible.
    rmax: float
        The maximum return value consideret to be feasible
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

    #2/3 gan generated goals and 1/3 of random goals. See Florensa et al. pp. 7
    old_goal_count = goal_count // 3
    gan_goal_count = 2*goal_count // 3
    _logger.debug(f"Using {gan_goal_count} GAN generated goals and {old_goal_count} old goals")
    
    #Save the mean value of the returns to plot a "coverage" graph
    avg_coverages = np.zeros((iter_count, ), dtype=np.float64)
    rewards = np.zeros((iter_count, ), dtype=np.float64)


    _logger.info(f"Starting initializing the GAN")
    _initialize_gan(gan, agent, env, gan_iter_count, goal_count, episode_count, timestep_count)
    _logger.info("Ending GAN initialization")
    
    
    #Count on how many consecutive iterations the agent cannot reach any of it's goals.
    consecutive_iters = 0
    
    old_goals = None
    for i in range(iter_count):
        _logger.info(f"OUTER ITERATION {i}")
    
        #---------- Create Goals -------------
        gan_goals = _create_goals(gan,  gan_goal_count)
        goals = gan_goals if old_goals is None else torch.cat([gan_goals, utils.sample_tensor(old_goals, old_goal_count)])
        env.goals = goals.detach().cpu().numpy()

        #---------- Update the policy ---------
        _logger.info(f"Starting updating the policy for {policy_iter_count} iterations")
        _update_or_eval_policy(agent,env, policy_iter_count, episode_count, timestep_count, eval_mode=False, global_step=i)
        _logger.info("Stopping the policy update")

        #---------- Evaluate the Policy -------
        _logger.info("Starting evaluating the policy")
        returns = _update_or_eval_policy(agent, env, 1, episode_count, timestep_count, eval_mode=True, global_step=i)
        _logger.info("Stopping evaluating the policy")
        
        avg_coverages[i] = np.mean(returns)

        #---------- Label goals ---------------
        labels = utils.label_goals(returns, rmin, rmax)

        if np.all(labels == 0):
            consecutive_iters += 1 
            _logger.warning(f"All labels 0 during training iteration {i+1}")  
        else:
            consecutive_iters = 0
        
        if consecutive_iters > 10:
            _logger.critical(f"[iter: {i}] {consecutive_iters} consecutive iters with 0-labels. Aborting!")
            break

        #---------- Train GAN ------------------
        gan.train(goals, labels, gan_iter_count, global_step=i + 1)

        #---------- Update Replay --------------
        old_goals = _update_replay(gan_goals, old_goals)
        

        utils.display_agent_and_goals(
            env.agent_pos, goals.detach().cpu().numpy(), returns, env.limits,
            rmin, rmax, filepath=f"images/goals_iter_{i}.png",
            pos_label="Agent position", title=f"Iteration {i}"
        )
        _logger.info(f"Displayed agent and goals")

        #---------- Save the models -------------
        if (i + 1) % save_after == 0:
            ddpg_path = utils.add_to_path(ddpg_base_path, f"iter_{i}")
            agent.save_model(ddpg_path)
            _logger.info(f"Saved DDPG to {ddpg_path}")

            gan_path = utils.add_to_path(gan_base_path, f"iter_{i}")
            gan.save_model(gan_path)
            _logger.info(f"Saved gan to {gan_path}")
    
    _clean_up(avg_coverages,rewards, gan, agent)
    _logger.info("Training done!")


