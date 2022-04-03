from models.agent import DDPGAgent
from environment.mujoco_env import MazeEnv
import numpy as np


def eval_policy(agent: DDPGAgent, env: MazeEnv, eval_iter: int, episode_count: int, timestep_count: int, render: bool = False) -> None:
    '''
    Evaluates performance of a policy and print out summary of those.

    Parameters
    ----------
    agent: DDPGAgent
        The policy/agent to evaluate
    env: MazeEnv
        The environment where the policy is evaluated.
    eval_iter: int
        The amount of iterations that the policy is evaluated for.
    episode_count: int
        The amount of episodes in each evaluation iteration.
    timestep_count: int
        The maximum amount of timesteps the agent has to reach the goal during an episode.
    render: bool, Optional
        If set to true, the environment will be rendered during each iteration. Default False.
    '''

    assert eval_iter > 0, f"The amount of evaluation iterations must be atleast 1, got {eval_iter}"
    
    header = f"{'evaluation':^16s}|{'episodes':^16s}|{'goal reached':^16s}|{'goals reached %':^16s}|{'avg timestep':^16s}"
    delim = f"{16*'-'}|{16*'-'}|{16*'-'}|{16*'-'}|{16*'-'}"
    values = "{:^16d}|{:^16d}|{:^16d}|{:^16.2f}%|{:^16.2f}"

    print(f"{header}\n{delim}")

    goals_reached = np.zeros(episode_count, dtype=np.int32)
    ts_counts = np.zeros(episode_count)

    env.eval = True
    env.goals = []

    for i in range(eval_iter):
        #Reset stats
        goals_reached.fill(False)
        ts_counts.fill(0.0)
        
        for ep in range(episode_count):
            state = env.reset()
            for ts in range(timestep_count):
                action = agent.act(state, use_noise=False)
                print(action)
                *_, done = env.step(action)
                if render:
                    env.render()
                if done:
                    ts_counts[ep] = ts
                    goals_reached[ep] = 1
                    break
        reached_count = np.count_nonzero(goals_reached)
        reached_idx = np.nonzero(goals_reached)
        print(values.format(i, episode_count, reached_count, float(reached_count/episode_count)*100,   np.mean(ts_counts[reached_idx])))
    print("Evaluation done!")




