import numpy as np
from src import agent

class AdaptiveDiscretization(agent.FiniteHorizonAgent):

    def __init__(self, epLen, numIters, scaling):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
        '''
        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling

        # List of tree's, one for each step
        self.tree_list = []
        # Fill with epLen number of trees

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        pass

        # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        pass

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        pass

    def split_ball(self, node):
        children = self.node.split_ball()

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        pass

    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action
