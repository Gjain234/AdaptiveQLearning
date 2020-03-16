import numpy as np
import tree
import time


class PendulumNode(tree.Node):
    # Splits a node by covering it with four children, as here S times A is [0,1]^2
    # each with half the radius
    def split_node(self):
        child_1 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] - self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_2 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] - self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        child_3 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] + self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_4 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] + self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        child_5 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] - self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_6 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] - self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        child_7 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] + self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_8 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] + self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        child_9 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] - self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_10 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] - self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        child_11 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] + self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_12 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2, self.state_val[2] + self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        child_13 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] - self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_14 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] - self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        child_15 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] + self.radius/2), self.action_val-self.radius/2, self.radius*(1/2))

        child_16 = PendulumNode(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2, self.state_val[2] + self.radius/2), self.action_val+self.radius/2, self.radius*(1/2))

        self.children = [child_1, child_2, child_3, child_4, child_5, child_6, child_7, child_8, child_9, child_10, child_11, child_12, child_13, child_14, child_15, child_16]
        return self.children


'''The tree class consists of a hierarchy of nodes for multiple ambulances'''
class PendulumTree(tree.Tree):
    # Defines a tree by the number of steps for the initialization
    def __init__(self, max_reward):
        self.head = PendulumNode(max_reward, 0, 0, (0.5, 0.5, 0.5), 0.5, 0.5)
        self.epLen = max_reward
        self.count = 0

    # def get_active_ball_recursion(self, state, node, step):
    #     # If the node doesn't have any children, then the largest one
    #     # in the subtree must be itself
    #     # print('Currently in node with state: ' + str(node.state_val) + ' and radius: ' + str(node.radius))
    #
    #     if node.children == None:
    #         return node, node.qVal
    #     else:
    #         # Otherwise checks each child node
    #         qVal = 0
    #         for child in node.children:
    #             # if the child node contains the current state
    #             if self.state_within_node(state, child):
    #                 # recursively check that node for the max one, and compare against all of them
    #                 new_node, new_qVal = self.get_active_ball_recursion(state, child, step)
    #                 if new_qVal < 0:
    #                     print(new_qVal)
    #                     print('ABORT SOMETHING BAD IS HAPPENING!!!!')
    #                     time.sleep(5)
    #                 if new_qVal >= qVal:
    #                     active_node, qVal = new_node, new_qVal
    #             else:
    #                 pass
    #     return active_node, qVal
    #
    # def get_active_ball(self, state, step):
    #     # print('Getting a new active ball for state: ' + str(state))
    #     active_node, qVal = self.get_active_ball_recursion(state, self.head, step)
    #     return active_node, qVal

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        #, np.abs(state[2] - node.state_val[2])
        return max(np.abs(state[0] - node.state_val[0]), np.abs(state[1] - node.state_val[1])) <= node.radius
