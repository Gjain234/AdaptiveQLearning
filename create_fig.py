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
nEps = 5000

problem_type = 'oil'
problem_list = ['quadratic', 'laplace']
param_list = ['1', '10', '50']
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams.update({'font.size': 8})

for problem in problem_list:
    for param in param_list:
        name_adapt = './data/oil_'+problem+'_'+'adapt_'+param+'.csv'
        name_net ='./data/oil_'+problem+'_'+'net_'+param+'.csv'
        name_obj = './data/oil_'+problem+'_agent_'+param+'.obj'
        fig_name = './figures/oil_'+problem+'_'+param+'.png'

        infile = open(name_obj,'rb')
        agent = pickle.load(infile)
        infile.close()

        dt_adapt = pd.read_csv(name_adapt).groupby(['episode']).mean()
        dt_net = pd.read_csv(name_net).groupby(['episode']).mean()
        dt_adapt['episode'] = dt_adapt.index.values
        dt_net['episode'] = dt_net.index.values
        dt_net = dt_net.iloc[::10, :]
        dt_adapt = dt_adapt.iloc[::10, :]

        fig = plt.figure(figsize=(7.2, 2.5))
        plt.subplot(1,3,1)

        plt.plot(dt_adapt['episode'], dt_adapt['epReward'], label='Adaptive')
        plt.plot(dt_net['episode'], dt_net['epReward'], label = 'Epsilon Net', linestyle='--')

        plt.ylim(0,epLen+.1)
        plt.xlabel('Episode')
        plt.ylabel('Observed Reward')
        plt.legend()
        plt.title('Comparison of Observed Rewards')

        plt.subplot(1,3,2)
        plt.plot(dt_adapt['episode'], dt_adapt['Number of Balls'])
        plt.plot(dt_net['episode'], dt_net['Number of Balls'], linestyle = '--')

        plt.xlabel('Episode')
        plt.ylabel('Size of Partition')
        plt.title('Comparison of Size of Partition')

        plt.subplot(1,3,3)
        tree = agent.tree_list[1]
        tree.plot(fig)
        plt.title('Adaptive Discretization for Step 2')
        plt.tight_layout()
        fig.savefig(fig_name, bbox_inches = 'tight',
            pad_inches = 0.01, dpi=900)


epLen = 5
nEps = 2000

problem_type = 'ambulance'
problem_list = ['uniform', 'beta']
param_list = ['0', '1', '25']


for problem in problem_list:
    for param in param_list:
        name_adapt = './data/ambulance_'+problem+'_'+'adapt_'+param+'.csv'
        name_net ='./data/ambulance_'+problem+'_'+'net_'+param+'.csv'
        name_median = './data/ambulance_'+problem+'_'+'median_'+param+'.csv'
        name_no = './data/ambulance_'+problem+'_'+'no_'+param+'.csv'
        name_obj = './data/ambulance_'+problem+'_agent_'+param+'.obj'
        fig_name = './figures/ambulance_'+problem+'_'+param+'.png'

        infile = open(name_obj,'rb')
        agent = pickle.load(infile)
        infile.close()

        dt_adapt = pd.read_csv(name_adapt).groupby(['episode']).mean()
        dt_net = pd.read_csv(name_net).groupby(['episode']).mean()
        dt_median = pd.read_csv(name_median).groupby(['episode']).mean()
        dt_no = pd.read_csv(name_no).groupby(['episode']).mean()
        dt_adapt['episode'] = dt_adapt.index.values
        dt_net['episode'] = dt_net.index.values
        dt_net = dt_net.iloc[::10, :]
        dt_adapt = dt_adapt.iloc[::10, :]
        dt_median['episode'] = dt_median.index.values
        dt_no['episode'] = dt_no.index.values
        dt_no = dt_no.iloc[::10, :]
        dt_median = dt_median.iloc[::10, :]


        fig = plt.figure(figsize=(7.2, 2.5))
        plt.subplot(1,3,1)
        plt.title('Comparison of Observed Rewards')
        plt.plot(dt_adapt['episode'], dt_adapt['epReward'], label='Adaptive')
        plt.plot(dt_net['episode'], dt_net['epReward'], label = 'Epsilon Net', linestyle='--')
        plt.plot(dt_median['episode'], dt_median['epReward'], label='Median', linestyle=':')
        plt.plot(dt_no['episode'], dt_no['epReward'], label = 'No Movement', linestyle='-.')

        plt.ylim(0,epLen+.1)
        plt.xlabel('Episode')
        plt.ylabel('Observed Reward')
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(dt_adapt['episode'], dt_adapt['Number of Balls'])
        plt.plot(dt_net['episode'], dt_net['Number of Balls'], linestyle = '--')

        plt.xlabel('Episode')
        plt.ylabel('Size of Partition')
        plt.title('Comparison of Size of Partition')

        plt.subplot(1,3,3)
        tree = agent.tree_list[1]
        tree.plot(fig)
        plt.title('Adaptive Discretization for Step 2')

        plt.tight_layout()
        fig.savefig(fig_name, bbox_inches = 'tight',
            pad_inches = 0.01, dpi=900)
