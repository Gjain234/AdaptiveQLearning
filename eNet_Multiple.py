import numpy as np
from src import agent

class eNet_Multiple(agent.FiniteHorizonAgent):

    def __init__(self, action_net, state_net, epLen, scaling):
        '''
        args:
            - action_net - epsilon net of action space
            - state_net - epsilon net of state space
            - epLen - steps per episode
            - scaling - scaling parameter for UCB terms
        '''
        self.action_net = action_net
        self.state_net = state_net
        self.epLen = epLen
        self.scaling = scaling

        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.state_net),len(self.action_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.state_net), len(self.action_net), len(self.action_net)], dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones([self.epLen, len(self.state_net), len(self.state_net),len(self.action_net), len(self.action_net)], dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen, len(self.state_net), len(self.state_net), len(self.action_net), len(self.action_net)], dtype=np.float32)

        '''
            Adds the observation to records by using the update formula
        '''
    def update_obs(self, obs, action, reward, newObs, timestep):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - obs[0])), np.argmin(np.abs(np.asarray(self.state_net) - obs[1])))
        action_discrete = (np.argmin(np.abs(np.asarray(self.action_net) - action[0])), np.argmin(np.abs(np.asarray(self.action_net) - action[1])))
        state_new_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - newObs[0])), np.argmin(np.abs(np.asarray(self.state_net) - newObs[1])))

        self.num_visits[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]] += 1
        t = self.num_visits[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]]
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)

        if timestep == self.epLen-1:
            vFn = 0
        else:
            vFn = max(self.qVals[timestep+1, state_new_discrete[0], state_new_discrete[1], :])
        vFn = min(self.epLen, vFn)

        self.qVals[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]] = (1 - lr) * self.qVals[timestep, state_discrete[0], state_discrete[1], action_discrete[0], action_discrete[1]] + lr * (reward + vFn + bonus)

    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net)**2 * len(self.action_net)**2

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # returns the discretized state location and takes action based on
        # maximum q value
        state_discrete = (np.argmin(np.abs(np.asarray(self.state_net) - state[0])), np.argmin(np.abs(np.asarray(self.state_net) - state[1])))
        qFn = self.qVals[timestep, state_discrete[0], state_discrete[1], :, :]
        
        # Get indices in qVals matrix for where the q function is maximized - should be two of them
        action_1, action_2 = np.where(qFn == qFn.max())
        index = np.random.choice(len(action_1))
        # from the two indices - return a tuple of the two actions for the action_net at those indices
        # self.action_net[i], self.action_net[j]
        return self.action_net[action_1[index]], self.action_net[action_2[index]] ##need to change!

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
