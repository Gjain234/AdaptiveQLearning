import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

''' Implementation of a tree structured used in the Adaptive Discretization Algorithm'''


''' First defines the node class by storing all relevant information'''
class Node():
    def __init__(self, qVal, num_visits, num_splits, state_val, action_val, radius):
        '''args:
        qVal - estimate of the q value
        num_visits - number of visits to the node or its ancestors
        num_splits - number of times the ancestors of the node has been split
        state_val - value of state at center
        action_val - value of action at center
        radius - radius of the node '''
        self.qVal = qVal
        self.num_visits = num_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = action_val
        self.radius = radius
        self.flag = False
        self.children = None

        # Splits a node by covering it with four children, as here S times A is [0,1]^2
        # each with half the radius
    def split_node(self):
        child_1 = Node(self.qVal, self.num_visits, self.num_splits+1, self.state_val+self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
        child_2 = Node(self.qVal, self.num_visits, self.num_splits+1, self.state_val+self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
        child_3 = Node(self.qVal, self.num_visits, self.num_splits+1, self.state_val-self.radius/2, self.action_val+self.radius/2, self.radius*(1/2))
        child_4 = Node(self.qVal, self.num_visits, self.num_splits+1, self.state_val-self.radius/2, self.action_val-self.radius/2, self.radius*(1/2))
        self.children = [child_1, child_2, child_3, child_4]
        return self.children

''' Implementation of a tree structured used in the Adaptive Discretization Algorithm for multiple ambulances'''


''' First defines the node class by storing all relevant information'''
class Node_Multiple_Ambulance():
    def __init__(self, qVal, num_visits, num_splits, state_val, action_val, radius):
        '''args:
        qVal - estimate of the q value
        num_visits - number of visits to the node or its ancestors
        num_splits - number of times the ancestors of the node has been split
        state_val - value of state at center
        action_val - value of action at center
        radius - radius of the node '''
        self.qVal = qVal
        self.num_visits = num_visits
        self.num_splits = num_splits
        self.state_val = (state_val[0],state_val[1])
        self.action_val = (action_val[0],action_val[1])
        self.radius = radius
        self.flag = False
        self.children = None

        # Splits a node by covering it with four children, as here S times A is [0,1]^2
        # each with half the radius
    def split_node(self):
        child_1 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_2 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        child_3 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_4 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        child_5 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_6 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        child_7 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_8 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]-self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        child_9 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_10 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        child_11 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_12 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]-self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        child_13 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_14 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]-self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        child_15 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]-self.radius/2), self.radius*(1/2))
        child_16 = Node_Multiple_Ambulance(self.qVal, self.num_visits, self.num_splits+1, (self.state_val[0]+self.radius/2, self.state_val[1]+self.radius/2), (self.action_val[0]+self.radius/2, self.action_val[1]+self.radius/2), self.radius*(1/2))
        
        
        self.children = [child_1, child_2, child_3, child_4, child_5, child_6, child_7, child_8, child_9, child_10, child_11, child_12, child_13, child_14, child_15, child_16]
        return self.children

'''The tree class consists of a hierarchy of nodes'''
class Tree():
    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen):
        self.head = Node(epLen, 0, 0, 0.5, 0.5, 0.5)
        self.epLen = epLen

    # Returns the head of the tree
    def get_head(self):
        return self.head

    # Plot function which plots the tree on a graph on [0,1]^2 with the discretization
    def plot(self, fig):
        ax = plt.gca()
        self.plot_node(self.head, ax)
        plt.xlabel('State Space')
        plt.ylabel('Action Space')
        return fig

    # Recursive method which plots all subchildren
    def plot_node(self, node, ax):
        if node.children == None:
            # print('Child Node!')
            rect = patches.Rectangle((node.state_val - node.radius,node.action_val-node.radius),node.radius*2,node.radius*2,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            # plt.text(node.state_val, node.action_val, np.around(node.qVal, 3))
        else:
            for child in node.children:
                self.plot_node(child, ax)


    # Recursive method which gets number of subchildren
    def get_num_balls(self, node):
        num_balls = 0
        if node.children == None:
            return 1
        else:
            for child in node.children:
                num_balls += self.get_num_balls(child)
        return num_balls

    def get_number_of_active_balls(self):
        return self.get_num_balls(self.head)


    # A method which implements recursion and greedily selects the selected ball
    # to have the largest qValue and contain the state being considered

    def get_active_ball_recursion(self, state, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        if node.children == None:
            return node, node.qVal
        else:
            # Otherwise checks each child node
            qVal = 0
            for child in node.children:
                # if the child node contains the current state
                if self.state_within_node(state, child):
                    # recursively check that node for the max one, and compare against all of them
                    new_node, new_qVal = self.get_active_ball_recursion(state, child)
                    if new_qVal >= qVal:
                        active_node, qVal = new_node, new_qVal
                else:
                    pass
        return active_node, qVal


    def get_active_ball(self, state):
        active_node, qVal = self.get_active_ball_recursion(state, self.head)
        return active_node, qVal

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        return np.abs(state - node.state_val) <= node.radius

'''The tree class consists of a hierarchy of nodes for multiple ambulances'''
class Tree_Multiple_Ambulance():
    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen):
        self.head = Node_Multiple_Ambulance(epLen, 0, 0, (0.5, 0.5), (0.5, 0.5), 0.5)
        self.epLen = epLen

    # Returns the head of the tree
    def get_head(self):
        return self.head

    # # Plot function which plots the tree on a graph on [0,1]^2 with the discretization
    # def plot(self, fig):
    #     ax = plt.gca()
    #     self.plot_node(self.head, ax)
    #     plt.xlabel('State Space')
    #     plt.ylabel('Action Space')
    #     return fig

    # # Recursive method which plots all subchildren
    # def plot_node(self, node, ax):
    #     if node.children == None:
    #         # print('Child Node!')
    #         rect = patches.Rectangle((node.state_val - node.radius,node.action_val-node.radius),node.radius*2,node.radius*2,linewidth=1,edgecolor='r',facecolor='none')
    #         ax.add_patch(rect)
    #         # plt.text(node.state_val, node.action_val, np.around(node.qVal, 3))
    #     else:
    #         for child in node.children:
    #             self.plot_node(child, ax)


    # Recursive method which gets number of subchildren
    def get_num_balls(self, node):
        num_balls = 0
        if node.children == None:
            return 1
        else:
            for child in node.children:
                num_balls += self.get_num_balls(child)
        return num_balls

    def get_number_of_active_balls(self):
        return self.get_num_balls(self.head)


    # A method which implements recursion and greedily selects the selected ball
    # to have the largest qValue and contain the state being considered

    def get_active_ball_recursion(self, state, node):
        # If the node doesn't have any children, then the largest one
        # in the subtree must be itself
        if node.children == None:
            return node, node.qVal
        else:
            # Otherwise checks each child node
            qVal = 0
            for child in node.children:
                # if the child node contains the current state
                if self.state_within_node(state, child):
                    # recursively check that node for the max one, and compare against all of them
                    new_node, new_qVal = self.get_active_ball_recursion(state, child)
                    if new_qVal >= qVal:
                        active_node, qVal = new_node, new_qVal
                else:
                    pass
        return active_node, qVal


    def get_active_ball(self, state):
        active_node, qVal = self.get_active_ball_recursion(state, self.head)
        return active_node, qVal

    # Helper method which checks if a state is within the node
    def state_within_node(self, state, node):
        return max(np.abs(state[0] - node.state_val[0]), np.abs(state[1] - node.state_val[1])) <= node.radius
