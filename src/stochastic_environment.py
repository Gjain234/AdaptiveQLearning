#-------------------------------------------------------------------------------
'''An multiple ambulance environment over [0,1].  An agent interacts through the environment
by picking a location to station the ambulance.  Then a patient arrives and the ambulance
most go and serve the arrival, paying a cost of travel.'''

class Stochastic_MultipleAmbulanceEnvironment(Environment):
    def __init__(self, epLen, arrivals, alpha, starting_state):
        '''
        epLen - number of steps
        arrivals - arrival distribution for patients
        alpha - parameter for difference in costs
        starting_state - starting locations
        '''
        self.epLen = epLen
        self.arrivals = arrivals
        self.alpha = alpha
        self.state = (starting_state[0], starting_state[1])
        self.starting_state = (starting_state[0], starting_state[1])
        self.timestep = 0


    def get_epLen(self):
        return self.epLen

    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = self.starting_state

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        old_state = self.state
        # new state is sampled from the arrivals distribution

        arrival = self.arrivals(self.timestep)
        reward = 1 - min(self.alpha*np.abs(old_state[0] - arrival) + (1 - self.alpha)*np.abs(arrival - action[0])), \
                self.alpha*np.abs(old_state[1] - arrival) + (1 - self.alpha)*np.abs(arrival - action[1])))

        newState = (action[0], action[1])

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1



        self.state = newState
        self.timestep += 1
        return reward, self.state, pContinue
#-------------------------------------------------------------------------------
