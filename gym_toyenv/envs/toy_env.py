import gym
import numpy as np
from gym import spaces


class ToyEnv(gym.Env):
    """
    Description:
        There are n levers, each with two states: on or off. All levers must be switched on to win the game. 
        You can only switch lever on if the lever on the left side of the lever is switched on. The leftmost lever
        can be switched on with no prequisites.

    Observation space:
        Type: Discrete(levers)
        Each index may have value of 0 or 1 representing the lever position.

    Action space:
        Type: Discrete(1)
        Each number from 0 to n represents an corresponding lever. 

    Reward:
        -1 each timestep except when reaching the terminal state

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, levers=5):
        """
        Init the environment.
        """
        super(ToyEnv, self).__init__()

        self.action_space = spaces.Discrete(levers)
        self.observation_space = spaces.Box(0, 1, (levers, ), dtype=np.int8)
        self.levers = levers
        self.state = np.array([0 for x in range(levers)])
        self.done = False
        self.reward = 0

    def step(self, action):
        """
        Environment takes a step forward. Takes action as input and outputs next state, 
        reward, if the game has been completed and info dictionary.
        """

        if action == 0 or self.state[action-1] == 1:
            self.state[action] = 1

        self.reward = -1

        if len(np.unique(self.state)) == 1 and self.state[0] != 0:
            self.reward = 0
            self.done = True

        return self.state, self.reward, self.done, {}

    def reset(self):
        """Reset the environment to start state."""
        self.state = np.array([0 for x in range(self.levers)])
        self.done = False
        self.reward = 0

        return self.state

    def render(self, mode='human'):
        """Render environment by printing it to terminal"""
        [print(x, end='') for x in self.state]
        print()