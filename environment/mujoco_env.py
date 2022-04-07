
import mujoco_maze #Must be imported for the maze enviroments
import gym
import numpy as np

from typing import Optional, Sequence, Tuple, Union
import numpy.typing as npt

import utils

_POS_IDX: int = 2 #The first two elements in a observation are the position of the agent.

class MazeEnv:
    '''
    Creates a Maze enviroment. This is done by adding a light wrapper to a OpenAI gym enviroment

    Parameters
    ----------
    env_name: str
        The name of the enviroment. Should be one of the OpenAI gym enviroments
    goal_size: Tuple[int] | int 
        The size of a single goal in the environment.
    initial_goals: Sequence[np.ndarray], Optional
        The initial goals for the enviroment. Default None
    tol: float, Optional
        The allowed tolerance, when a goal is counted as reached. Default 1e-2
    reward_range: Tuple[float, float], Optional
        The range, where the results should be. Default (0.0, 1.0)
    '''

    def __init__(self, env_name: str, goal_size: Union[Tuple[int], int], initial_goals: Optional[Sequence[npt.ArrayLike]] = None, tol: float = 1e-2, reward_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self._env: gym.Env = gym.make(env_name)
        self._agent_pos: Optional[npt.NDArray[np.float32]] = None
        
        self._goals: Optional[Sequence[npt.ArrayLike]] = initial_goals
        self._achieved_goals: Optional[Sequence[bool]] = None
        self._achieved_goals_counts: Optional[npt.NDArray[np.float32]] = None
        
        self._eval: bool = False

        self._tol: float = tol
        self._reward_range: Tuple[float, float] = reward_range
        self._goal_size: int = goal_size if isinstance(goal_size, int) else goal_size[0] 

        self._logger = utils.get_logger(__name__)

        self._initialize()
    
    def _initialize(self) -> None:
        '''
        Initializes the enviroment, by reseting the enviroment, 
        and taking an action
        '''
        obs = self._env.reset()
        self._agent_pos = obs[:_POS_IDX].copy()
        assert self._agent_pos is not None, "The agent position was not set!"

    def _is_goal_reached(self, goal: npt.NDArray) -> bool:
        '''
        Checks if a given goal is reached based on
        euclidian distance.
        
        Parameters
        ----------
        goal: npt.NDArray
            The goal that might be reached.
        
        Returns
        -------
        True if the distance between the current position and the goal is small enough (defined by tol), false otherwise

        '''
        return np.linalg.norm(self._agent_pos - goal) < self._tol 

    def _get_reward(self) -> float:
        '''Reward is a negative distance to the closest goal'''
        return -np.linalg.norm(self._agent_pos - self._goals, axis=-1).mean()

    @property
    def action_space(self) -> gym.spaces.box.Box:
        ''' Returns the action space of the environment '''
        return self._env.action_space

    @property
    def observation_space(self) -> gym.spaces.box.Box:
        '''Returns the observation space of the environment '''
        return self._env.observation_space
    
    @property
    def obs_limits(self) -> Tuple[float, float]:
        '''Return the limits of the environment in (low, high) tuple'''
        low = self.observation_space.low.max()
        high = self.observation_space.high.min()
        return low, high
    
    @property
    def action_limits(self) -> Tuple[float, float]:
        '''Returns the limts of the action space in (low, high) tuple'''
        low = self.action_space.low.max()
        high = self.action_space.high.min()
        return low, high
    
    @property
    def eval(self) -> bool:
        '''Returns the current evaluation state'''
        return self._eval

    @eval.setter
    def eval(self, val: bool) -> None:
        '''
        Sets if the environment is in evaluation mode or not.
        (e.g. does the environment track which goals are reached)

        Parameter
        ---------
        val: bool
            The new value. If true, the environment tracks which 
            goals are reached and which not.
        '''
        self._eval = val

    @property
    def goals(self) -> Sequence[npt.ArrayLike]:
        ''' Returns the current goals'''
        return self._goals

    @goals.setter
    def goals(self, new_goals: Sequence[npt.ArrayLike]) -> None:
        '''
        Sets the current goals for the agent, and 
        resets the statistics of the goals

        Parameters
        ----------
        new_goals: Sequence[npt.NDArray]
            The new goals to use in the enviroment
        '''
        self._goals = new_goals
        self._achieved_goals = np.zeros( (len(self._goals), ), dtype=bool)
        self._achieved_goals_counts = np.zeros ( len(self._goals, ), dtype=np.float32)

        self._logger.debug("Set new goals, and reseted achieved goals")


    @property
    def agent_pos(self) -> npt.NDArray:
        '''Returns the current position of the agent'''
        assert self._agent_pos is not None, "The position of the agent wasn't set!"
        return self._agent_pos

    @property
    def achieved_goals(self) -> Sequence[bool]:
        ''' Returns a list specifying which of the goals where achieved'''
        return self._achieved_goals

    @property
    def achieved_goals_counts(self) -> npt.NDArray[np.float32]:
        ''' Returns an array specifying how many times each goal was reached'''
        return self._achieved_goals_counts

    @property
    def goal_size(self) -> int:
        '''Returns the size of the goals'''
        return self._goal_size

    def step(self, action: npt.NDArray) -> Tuple[npt.NDArray, float, bool]:
        '''
        Takes a step in the current enviroment, and evaluates if 
        any of the current goals was reached. 

        Parameters
        ----------
        action: np.NDArray
            The action to take from the current state
        Returns
        -------
        Tuple[np.ndarray, float, bool]
            Returns a tuple, that consists of an observation, reward and information about if (any) goal
            was reached.
        '''

        #The reward, and done given by the enviroment are ignored.
        obs, _ ,_ , info = self._env.step(action)

        self._agent_pos = info.get("position", None)
        goal_reached = False

        assert self._agent_pos is not None, "No position was found for the agent"
        if self._goals is None:
            raise RuntimeError("The step was tried to call before the goals were set!")

        for i, g in enumerate(self._goals):
            
            assert g.shape == self._agent_pos.shape, "The shapes of the goals and agent positions don't match"
            if self._is_goal_reached(g):
                self._logger.debug("A goal was found in Maze")
                goal_reached = True
                
                if self._eval:
                    self._logger.debug("Eval mode: add goal statistics")
                    self._achieved_goals[i] = True
                    self._achieved_goals_counts[i] += 1

        #reward = 1.0 if goal_reached else 0.0
        return obs, self._get_reward(),  goal_reached

    def reset(self) -> npt.NDArray:
        '''
        Resets the current enviroment. (But not goals, nor their statistics) 
 
        Returns
        -------
        npt.NDArray
            The observation resulted from the reset
        '''
        obs = self._env.reset()
        self._agent_pos = obs[:_POS_IDX].copy()
        #self._logger.info("Environment reseted")
        assert self._agent_pos is not None, "The agent position was set to None!"
        return obs


    def render(self,  **kwargs) -> None:
        ''' Renders the enviroment '''
        return self._env.render(**kwargs)

    def close(self) -> None:
        ''' Closes the current enviroment '''
        self._env.close()


