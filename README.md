# Adaptive Discretization for Reinforcement Learning
This repository contains a reference implementation for the algorithm
appearing in the paper \[2\] for model-free Q learning in continuous spaces.

### Dependencies
The code has been tested in `Python 3.6.5` and depends on a number of Python
packages. For the core implementation, found under `src/`:

* `environment.py`: defines an environment the agent interacts in
* `agent.py`: defines the agent
* `experiment.py`: defines an experiment and saves data

These implementations are adapted from TabulaRL developed by Ian Osband \[1\] extended to continuous state action spaces.


For the remaining scripts which are aimed to reproduce some of the experimental
results found in the paper and can be found in the root directory of this repo,
the following packages are required:

* numpy
* matplotlib
* pandas
* seaborn

### Quick Tour

We offer implementations for adaptive discretization from \[2\] and the epsilon net algorithm from \[3\].  Both algorithms are set up to run with state space [0,1] and action space [0,1] but the extension to other state and action spaces should be similar.

The following files implement the different algorithms:
* `adaptive_Agent.py`: implements the Adaptive Discretization algorithm
* `eNet_Agent.py`: implements the discrete algorithm on the epsilon net
* `data_Agent.py`: implements the heuristic algorithms discussed for the ambulance problem

These agents are imported and used in the different tests.  To run the experiments used in the paper the following three files can be used.
* `test_environment.py`
* `oil_environment.py`
* `ambulance.environment.py`

Each file has parameters at the top which can be changed in order to replicate the parameters considered for each experiment in the paper.


\[1\]: Ian Osband, TabulaRL (2017), Github Repository. https://github.com/iosband/TabulaRL

\[2\]: Sean R. Sinclair, Siddhartha Banerjee, Christina Yu. *Adaptive Discretization for Episodic Reinforcement Learning in Metric Spaces.*
Available [coming soon]().

\[3\]: Zhao Song, Wen Sun. *Efficient Model-free Reinfrocement Learning in Metric Spaces.* Available [here](https://arxiv.org/abs/1905.00475).
