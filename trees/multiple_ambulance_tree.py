import numpy as np
import tree

''' Implementation of a tree structured used in the Adaptive Discretization Algorithm for multiple ambulances'''

''' First defines the node class by storing all relevant information'''


class MultipleAmbulanceNode(tree.Node):
    def split_node(self):
        child_1 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_2 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_3 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_4 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_5 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_6 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_7 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_8 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] - self.radius / 2, self.state_val[1] + self.radius / 2),
                                          (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                          self.radius * (1 / 2))
        child_9 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                          (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                          (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                          self.radius * (1 / 2))
        child_10 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                           (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))
        child_11 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                           self.radius * (1 / 2))
        child_12 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] - self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))
        child_13 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] - self.radius / 2, self.action_val[1] - self.radius / 2),
                                           self.radius * (1 / 2))
        child_14 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] - self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))
        child_15 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] - self.radius / 2),
                                           self.radius * (1 / 2))
        child_16 = MultipleAmbulanceNode(self.qVal, self.num_visits, self.num_splits + 1,
                                           (self.state_val[0] + self.radius / 2, self.state_val[1] + self.radius / 2),
                                           (self.action_val[0] + self.radius / 2, self.action_val[1] + self.radius / 2),
                                           self.radius * (1 / 2))

        self.children = [child_1, child_2, child_3, child_4, child_5, child_6, child_7, child_8, child_9, child_10,
                         child_11, child_12, child_13, child_14, child_15, child_16]
        return self.children


'''The tree class consists of a hierarchy of nodes for multiple ambulances'''


class MultipleAmbulanceTree(tree.Tree):
    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen):
        self.head = MultipleAmbulanceNode(epLen, 0, 0, (0.5, 0.5), (0.5, 0.5), 0.5)
        self.epLen = epLen

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        return max(np.abs(state[0] - node.state_val[0]), np.abs(state[1] - node.state_val[1])) <= node.radius
