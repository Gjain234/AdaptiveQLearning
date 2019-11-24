import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import pickle
import numpy as np
from src import agent
from adaptive_Agent import AdaptiveDiscretization
from eNet_Agent import eNet
import pandas as pd

epLen = 5
nEps = 2000

problem_type = 'ambulance'
problem_list = ['uniform']
alpha_list = ['0', '0.25', '1']
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams.update({'font.size': 8})

for alpha in alpha_list:
    for problem in problem_list:
        name_adapt = './data/multiple_ambulance_'+problem+'_adapt_'+ alpha +'.csv'
        name_adapt_stochastic ='./data/multiple_ambulance_'+problem+'_adapt_stochastic_'+ alpha +'.csv'
        name_net = './data/multiple_ambulance_'+problem+'_enet_'+ alpha +'.csv'
        name_net_stochastic = './data/multiple_ambulance_'+problem+'_enet_stochastic_'+ alpha +'.csv'
        #name_obj = './data/multiple_ambulance_'+problem+'_agent_compare.obj'
        fig_name = './figures/multiple_ambulance_'+problem+'_'+alpha +'.png'

        #infile = open(name_obj,'rb')
        #agent = pickle.load(infile)
        #infile.close()


        dt_adapt = pd.read_csv(name_adapt).groupby(['episode']).mean()
        dt_net = pd.read_csv(name_net).groupby(['episode']).mean()
        dt_adapt_stochastic = pd.read_csv(name_adapt_stochastic).groupby(['episode']).mean()
        dt_net_stochastic = pd.read_csv(name_net_stochastic).groupby(['episode']).mean()
        #print(dt_adapt.index.values)
        dt_adapt['episode'] = dt_adapt.index.values
        dt_net['episode'] = dt_net.index.values
        dt_adapt_stochastic['episode'] = dt_adapt_stochastic.index.values
        dt_net_stochastic['episode'] = dt_net_stochastic.index.values

        dt_net = dt_net.iloc[::10, :]
        dt_net_stochastic = dt_net_stochastic.iloc[::10, :]
        dt_adapt = dt_adapt.iloc[::10, :]
        dt_adapt_stochastic = dt_adapt_stochastic.iloc[::10, :]

        fig = plt.figure(figsize=(10, 10))

        # Plot for Comparison of Observed Rewards of Adaptive vs E-Net
        plt.subplot(2,2,1)
        plt.plot(dt_adapt['episode'], dt_adapt['epReward'], label='Adaptive')
        plt.plot(dt_net['episode'], dt_net['epReward'], label = 'Epsilon Net', linestyle='--')

        plt.ylim(0,epLen+.1)
        plt.xlabel('Episode')
        plt.ylabel('Observed Reward')
        plt.legend()
        plt.title('Comparison of Observed Rewards of Adaptive vs E-Net')

        # Plot for Comparison of Size of Partition of Adaptive vs E-Net
        plt.subplot(2,2,2)
        plt.plot(dt_adapt['episode'], dt_adapt['Number of Balls'])
        plt.plot(dt_net['episode'], dt_net['Number of Balls'], linestyle = '--')

        plt.xlabel('Episode')
        plt.ylabel('Size of Partition')
        plt.legend()
        plt.title('Comparison of Size of Partition of Adaptive vs E-Net')

        # Plot for Comparison of Observed Rewards of Adaptive vs E-Net (Stochastic)
        plt.subplot(2,2,3)

        plt.plot(dt_adapt_stochastic['episode'], dt_adapt_stochastic['epReward'], label='Adaptive')
        plt.plot(dt_net_stochastic['episode'], dt_net_stochastic['epReward'], label = 'Epsilon Net', linestyle='--')

        plt.ylim(0,epLen+.1)
        plt.xlabel('Episode')
        plt.ylabel('Observed Reward')
        plt.legend()
        plt.title('Comparison of Observed Rewards of Adaptive vs E-Net (Stochastic)')

        # Plot for Comparison of Size of Partition of Adaptive vs E-Net (Stochastic)
        plt.subplot(2,2,4)
        plt.plot(dt_adapt_stochastic['episode'], dt_adapt_stochastic['Number of Balls'])
        plt.plot(dt_net_stochastic['episode'], dt_net_stochastic['Number of Balls'], linestyle = '--')

        plt.xlabel('Episode')
        plt.ylabel('Size of Partition')
        plt.legend()
        plt.title('Comparison of Size of Partition of Adaptive vs E-Net (Stochastic)')


        #plt.subplot(1,3,3)
        #tree = agent.tree_list[1]
        #tree.plot(fig)
        #plt.title('Adaptive Discretization for Step 2')
        plt.tight_layout()
        fig.savefig(fig_name, bbox_inches = 'tight',
            pad_inches = 0.01, dpi=900)
