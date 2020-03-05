import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from agents import pendulum_agent
from eNet_Agent import eNet
from src import environment
from src import experiment
from src import agent


''' Defining parameters to be used in the experiment'''
epLen = 5
nEps = 1000
numIters = 2





env = environment.make_pendulumEnvironment(epLen,False)

print(1)
scaling = 0.5
agent_list_adap = []
for _ in range(numIters):
    agent_list_adap.append(pendulum_agent.PendulumAgent(epLen, nEps, scaling))

dict = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 1, 'numIters' : numIters}

exp = experiment.Experiment(env, agent_list_adap, dict)
exp.run()
dt_adapt = exp.save_data()
print(2)
epsilon = (nEps * epLen)**(-1 / 4)
action_net = np.arange(start=0, stop=1, step=epsilon)
state_net = np.arange(start=0, stop=1, step=epsilon)
scaling = 0.5

agent_list = []
for _ in range(numIters):
    agent_list.append(eNet(action_net, state_net, epLen, scaling))

print(3)
exp = experiment.Experiment(env, agent_list, dict)
exp.run()
dt_net = exp.save_data()
print(4)
fig = plt.figure()
plt.subplot(1,2,1)
plt.errorbar(dt_adapt['episode'], dt_adapt['epReward'], label = 'Adaptive Zooming')

plt.errorbar(dt_net['episode'], dt_net['epReward'], label = 'Epsilon Net')

plt.xlabel('Episode')
plt.ylabel('Expected Reward')

plt.subplot(1,2,2)
adapt_line = plt.errorbar(dt_adapt['episode'], dt_adapt['Number of Balls'], label='Adaptive Zooming')


epsilon_line = plt.plot(dt_net['episode'], dt_net['Number of Balls'], label='Epsilon Net')

plt.xlabel('Episode')
plt.ylabel('Number of Active Balls')
plt.legend()
plt.show()

agent = agent_list_adap[-1]
print(agent.get_num_arms())
for tree in agent.tree_list:
    print(tree.get_number_of_active_balls())
    tree.plot()
