import numpy as np
import tree

''' Implementation of a tree structured used in the Adaptive Discretization Algorithm'''


''' First defines the node class by storing all relevant information'''


class AmbulanceNode(tree.Node):
    # Splits a node by covering it with four children, as here S times A is [0,1]^2
    # each with half the radius
    def split_node(self):
        child_1 = AmbulanceNode(self.qVal, self.num_visits, self.num_splits+1, self.state_val+self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
        child_2 = AmbulanceNode(self.qVal, self.num_visits, self.num_splits+1, self.state_val+self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
        child_3 = AmbulanceNode(self.qVal, self.num_visits, self.num_splits+1, self.state_val-self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
        child_4 = AmbulanceNode(self.qVal, self.num_visits, self.num_splits+1, self.state_val-self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
        self.children = [child_1, child_2, child_3, child_4]
        return self.children


'''The tree class consists of a hierarchy of nodes'''

class AmbulanceTree(tree.Tree):
    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen):
        self.head = AmbulanceNode(epLen, 0, 0, 0.5, 0.5, 0.5)
        self.epLen = epLen

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        return np.abs(state - node.state_val) <= node.radius
