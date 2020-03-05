from tree import *
import numpy as np
from agents import ambulance_agent
from trees import multiple_ambulance_tree

class MultipleAmbulanceAgent(ambulance_agent.AmbulanceAgent):

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

        # Makes a new partition for each step and adds it to the list of trees
        for h in range(epLen):
            tree = multiple_ambulance_tree.MultipleAmbulanceTree(epLen)
            self.tree_list.append(tree)

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        self.tree_list = []
        for h in range(self.epLen):
            tree = multiple_ambulance_tree.MultipleAmbulanceTree(self.epLen)
            self.tree_list.append(tree)

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
        tree = self.tree_list[timestep]

        # Gets the selected ball
        active_node, qVal = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action_one = np.random.uniform(active_node.action_val[0] - active_node.radius, active_node.action_val[0] + active_node.radius)
        action_two = np.random.uniform(active_node.action_val[1] - active_node.radius, active_node.action_val[1] + active_node.radius)

        return action_one, action_two