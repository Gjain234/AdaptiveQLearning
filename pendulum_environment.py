import gym
import numpy as np
from agents import pendulum_agent
from eNet_Agent import eNet
from eNet_Agent import eNet_Discount
from eNet_Agent import eNetPendulum
from src import environment
from src import experiment
from src import agent
import pickle


''' Defining parameters to be used in the experiment'''

epLen = 200
nEps = 2000
numIters = 50


env = environment.make_pendulumEnvironment(epLen, False)

##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT

scaling_list = [0.04, 0.035, 0.045]
# scaling_list = [5, 0.6]
#scaling_list = [0.04, 0.5]
epsilon_list = [0.1]
# scaling_list = [0.5, .01] # alpha = 1
# scaling_list = [1, .4] # alpha = 0
# scaling_list = [0.5, 0.01] # alpha = 0.25
max_reward_adapt = 0
max_reward_e_net = 0
opt_adapt_scaling = 0.01
opt_e_net_scaling = 0.01
count = 0


# TRYING OUT EACH SCALING FACTOR FOR OPTIMAL ONE
for scaling in scaling_list:
    for epsilon in epsilon_list:
        experiment_dict = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters}
        #

        # RUNNING EXPERIMENT FOR ADAPTIVE ALGORITHM


        agent_list_adap = []
        for _ in range(numIters):
            agent_list_adap.append(pendulum_agent.PendulumAgent(epLen, nEps, scaling, 0.995))
        exp = experiment.Experiment(env, agent_list_adap, experiment_dict)

        adap_fig = exp.run()
        dt_adapt_data = exp.save_data()

        if (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0] > max_reward_adapt:
            max_reward_adapt = (dt_adapt_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
            opt_adapt_scaling = scaling
            dt_adapt = dt_adapt_data
            opt_adapt_agent_list = agent_list_adap

        del agent_list_adap
        del dt_adapt_data

        # RUNNING EXPERIMENT FOR EPSILON NET ALGORITHM

        # action_net = np.arange(start=0, stop=1, step=epsilon)
        # state_net = np.arange(start=0, stop=1, step=epsilon)
        #
        # agent_list = []
        # for _ in range(numIters):
        #     agent_list.append(eNet_Discount(action_net, state_net, 0.99, scaling, (3,1)))
        #
        # exp = experiment.Experiment(env, agent_list, experiment_dict, save=True)
        # exp.run()
        # dt_net_data = exp.save_data()
        #
        # curr_reward = (dt_net_data.groupby(['episode']).mean().tail(1))['epReward'].iloc[0]
        # if curr_reward > max_reward_e_net:
        #     max_reward_e_net = curr_reward
        #     opt_e_net_scaling = scaling
        #     opt_epsilon_scaling = epsilon
        #     dt_net = dt_net_data
        #
        # del agent_list
        # del dt_net_data

#print(opt_adapt_scaling)
#print(opt_epsilon_scaling)
#print(opt_e_net_scaling)



# SAVING DATA TO CSV

#dt_adapt.to_csv('pendulum_adapt.csv')
#dt_net.to_csv('pendulum_net_1.csv')