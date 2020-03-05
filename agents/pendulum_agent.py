import adaptive_Agent
from tree import *
import numpy as np
from trees import pendulum_tree

class PendulumAgent(adaptive_Agent.AdaptiveDiscretization):

    def __init__(self, epLen, numIters, scaling, discount_factor):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
        '''
        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling
        self.discount_factor = discount_factor
        # Single tree for all of the steps
        self.tree = pendulum_tree.PendulumTree(1/(1-self.discount_factor))

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        self.tree = pendulum_tree.PendulumTree(1/(1-self.discount_factor))

    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''
        # Gets the active tree based on current timestep
        tree = self.tree
        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        if timestep == self.epLen - 1:
            vFn = 0
        else:
            # Gets the next tree to get the approximation to the value function
            # at the next timestep
            new_tree = self.tree
            new_active, new_q = new_tree.get_active_ball(newObs)
            vFn = min(1/(1-self.discount_factor), new_q)
        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + self.discount_factor*vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 4**active_node.num_splits and active_node.num_splits <= 10:
            active_node.split_node()

    # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        return self.tree.get_number_of_active_balls()

    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree

        # Gets the selected ball
        active_node, qVal = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action = np.random.uniform(active_node.action_val - active_node.radius, active_node.action_val + active_node.radius)

        return action
