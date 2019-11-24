import gym
import numpy as np
from adaptive_Agent import AdaptiveDiscretization, AdaptiveDiscretizationMultiple
from eNet_Agent import eNet_Multiple
from data_Agent import dataUpdateAgent
from src import environment
from src import multiple_ambulance_experiment
from src import agent
import pickle


''' Defining parameters to be used in the experiment'''

epLen = 5
nEps = 2000
numIters = 50
loc = 0.8+np.pi/60
scale = 2
def arrivals(step):
    return np.random.uniform(0,1)

alpha = [0, 0.25, 1]
scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
for a in alpha:
    starting_state = (0.5,0.5)

    env1 = environment.make_ambulanceEnvMDP_multiple(epLen, arrivals, a, starting_state)
    env2 = environment.make_ambulanceEnvMDP_stochastic(epLen, arrivals, a, starting_state)

    ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT
    
    max_reward_adapt = 0
    max_reward_e_net = 0
    max_reward_adapt_stochastic = 0
    max_reward_e_net_stochastic = 0
    opt_adapt_scaling = 0.01
    opt_e_net_scaling = 0.01
    opt_adapt_stochastic_scaling = 0.01
    opt_e_net_stochastic_scaling = 0.01

    # TRYING OUT EACH SCALING FACTOR FOR OPTIMAL ONE

    for scaling in scaling_list:

        # RUNNING EXPERIMENT FOR ADAPTIVE ALGORITHM 

        agent_list_adap = []
        for _ in range(numIters):
            agent_list_adap.append(AdaptiveDiscretizationMultiple(epLen, nEps, scaling))

        dict = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters}

        exp = multiple_ambulance_experiment.Experiment(env1, agent_list_adap, dict)
        adap_fig = exp.run()
        dt_adapt_data = exp.save_data()

        if (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_adapt:
            max_reward_adapt = (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
            opt_adapt_scaling = scaling
            dt_adapt = dt_adapt_data
            opt_adapt_agent_list = agent_list_adap

        # RUNNING EXPERIMENT FOR ADAPTIVE ALGORITHM - STOCHASTIC VERSION
        agent_list_adap_stochastic = []
        for _ in range(numIters):
            agent_list_adap_stochastic.append(AdaptiveDiscretizationMultiple(epLen, nEps, scaling))

        exp = multiple_ambulance_experiment.Experiment(env2, agent_list_adap_stochastic, dict)
        adap_fig_stochastic = exp.run()
        dt_adapt_stochastic_data = exp.save_data()

        if (dt_adapt_stochastic_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_adapt_stochastic:
            max_reward_adapt_stochastic = (dt_adapt_stochastic_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
            opt_adapt_stochastic_scaling = scaling
            dt_adapt_stochastic = dt_adapt_stochastic_data
            opt_adapt_stochastic_agent_list = agent_list_adap_stochastic

        # RUNNING EXPERIMENT FOR EPSILON NET ALGORITHM REGULAR   
        epsilon = (nEps * epLen)**(-1 / 4)
        action_net = np.arange(start=0, stop=1, step=epsilon)
        state_net = np.arange(start=0, stop=1, step=epsilon)

        agent_list = []
        for _ in range(numIters):
            agent_list.append(eNet_Multiple(action_net, state_net, epLen, scaling))

        exp = multiple_ambulance_experiment.Experiment(env1, agent_list, dict)
        exp.run()
        dt_net_data = exp.save_data()

        if (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_e_net:
            max_reward_e_net = (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
            opt_e_net_scaling = scaling
            dt_net = dt_net_data

        # RUNNING EXPERIMENT FOR EPSILON NET ALGORITHM STOCHASTIC   

        exp = multiple_ambulance_experiment.Experiment(env2, agent_list, dict)
        exp.run()
        dt_net_stochastic_data = exp.save_data()

        if (dt_net_stochastic_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_e_net_stochastic:
            max_reward_e_net_stochastic = (dt_net_stochastic_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
            opt_e_net_stochastic_scaling = scaling
            dt_net_stochastic = dt_net_stochastic_data
    dt_adapt.to_csv('data/multiple_ambulance_uniform_adapt_'+ str(a) + '.csv')
    dt_adapt_stochastic.to_csv('data/multiple_ambulance_uniform_adapt_stochastic_'+ str(a) + '.csv')
    dt_net.to_csv('data/multiple_ambulance_uniform_enet_'+ str(a) + '.csv')
    dt_net_stochastic.to_csv('data/multiple_ambulance_uniform_enet_stochastic_'+ str(a) + '.csv')
#print(opt_adapt_scaling)
#print(opt_adapt_stochastic_scaling)


# SAVING DATA TO CSV


#agent = opt_adapt_agent_list[-1]
#filehandler = open('multiple_ambulance_uniform_agent_2.obj', 'wb')
#pickle.dump(agent, filehandler)
