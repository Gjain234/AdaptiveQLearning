import gym
import numpy as np
from adaptive_Agent import AdaptiveDiscretization
from eNet_Agent import eNet
from data_Agent import dataUpdateAgent
from src import environment
from src import experiment
from src import agent
import pickle


''' Defining parameters to be used in the experiment'''

epLen = 5
nEps = 2000
numIters = 50
loc = 0.8+np.pi/60
scale = 2
def arrivals():
    return np.random.uniform(0,1)

alpha = 1
starting_state = 0.5

env = environment.make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state)


##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT

scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
# scaling_list = [0.25, .01] # alpha = 0.25
# scaling_list = [0.25, .1] # alpha = 1
# scaling_list = [0.1] # alpha = 0
max_reward_adapt = 0
max_reward_e_net = 0
opt_adapt_scaling = 0.01
opt_e_net_scaling = 0.01

# TRYING OUT EACH SCALING FACTOR FOR OPTIMAL ONE
for scaling in scaling_list:

    # RUNNING EXPERIMENT FOR ADAPTIVE ALGORITHM

    agent_list_adap = []
    for _ in range(numIters):
        agent_list_adap.append(AdaptiveDiscretization(epLen, nEps, scaling))

    dict = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters}

    exp = experiment.Experiment(env, agent_list_adap, dict)
    adap_fig = exp.run()
    dt_adapt_data = exp.save_data()

    if (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_adapt:
        max_reward_adapt = (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
        opt_adapt_scaling = scaling
        dt_adapt = dt_adapt_data
        opt_adapt_agent_list = agent_list_adap

    # RUNNING EXPERIMENT FOR EPSILON NET ALGORITHM

    epsilon = (nEps * epLen)**(-1 / 4)
    action_net = np.arange(start=0, stop=1, step=epsilon)
    state_net = np.arange(start=0, stop=1, step=epsilon)

    agent_list = []
    for _ in range(numIters):
        agent_list.append(eNet(action_net, state_net, epLen, scaling))

    exp = experiment.Experiment(env, agent_list, dict)
    exp.run()
    dt_net_data = exp.save_data()

    if (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_e_net:
        max_reward_e_net = (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
        opt_e_net_scaling = scaling
        dt_net = dt_net_data

print(opt_adapt_scaling)
print(opt_e_net_scaling)


# RUNNING THE MEDIAN HEURISTIC ALGORITHM

input('Heuristic Agents')

# Heuristic agent
agent_list = []
for _ in range(numIters):
    agent_list.append(dataUpdateAgent(epLen, np.median, alpha))

exp = experiment.Experiment(env, agent_list, dict)
exp.run()
dt_median = exp.save_data()

# RUNNING THE NO MOVEMENT HEURISTIC ALGORITHM

# Don't Move Agent
def no_move(l):
    return l[-1]

agent_list = []
for _ in range(numIters):
    agent_list.append(dataUpdateAgent(epLen, no_move, alpha))
exp = experiment.Experiment(env, agent_list, dict)
exp.run()
dt_no_move = exp.save_data()

# SAVING DATA TO CSV

dt_adapt.to_csv('ambulance_uniform_adapt_1.csv')
dt_net.to_csv('ambulance_uniform_net_1.csv')
dt_median.to_csv('ambulance_uniform_median_1.csv')
dt_no_move.to_csv('ambulance_uniform_no_1.csv')
agent = opt_adapt_agent_list[-1]
filehandler = open('ambulance_uniform_agent_1.obj', 'wb')
pickle.dump(agent, filehandler)
