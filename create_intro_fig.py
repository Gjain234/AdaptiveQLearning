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
problem_list = ['quadratic']
param_list = ['50']
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rcParams.update({'font.size': 8})

for problem in problem_list:
    for param in param_list:
        name_adapt = './data/oil_'+problem+'_'+'adapt_'+param+'.csv'
        name_net ='./data/oil_'+problem+'_'+'net_'+param+'.csv'
        name_obj = './data/oil_'+problem+'_agent_'+param+'.obj'
        fig_name = './data/oil_'+problem+'_'+param+'.png'

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
        plt.title('Comparison of Observed Rewards')
        plt.legend()

        epsilon = (nEps * epLen)**(-1 / 4)
        action_net = np.arange(start=0, stop=1, step=epsilon)
        state_net = np.arange(start=0, stop=1, step=epsilon)

        plt.subplot(1,3,2)
        ax = plt.gca()
        for s in state_net:
            for a in action_net:
                rect = patches.Rectangle((s,a),epsilon, epsilon,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
        plt.xlabel('State Space')
        plt.ylabel('Action Space')
        plt.title('Uniform Discretization')

        plt.subplot(1,3,3)
        tree = agent.tree_list[1]
        tree.plot(fig)
        plt.title('Adaptive Discretization')

        plt.tight_layout()
        fig.savefig('./figures/intro_fig.png', bbox_inches = 'tight',
            pad_inches = 0.01, dpi=1200)
