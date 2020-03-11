from src import environment
import pickle


def simulate(file, numIter):
    agent = pickle.load(open(file, "rb"))
    env = environment.make_pendulumEnvironment(200, True)
    env.reset()
    oldState = env.state
    h = 0
    for i in range(numIter):
        env.reset()
        epReward = 0
        oldState = env.state
        pContinue = 1
        h=0
        while h < env.epLen and pContinue>0:
            # Step through the episode
            action = agent.pick_action(oldState, h)
            reward, newState, pContinue = env.advance(action)
            epReward+=reward
            #agent.update_obs(oldState, action, reward, newState, h)
            oldState = newState
            h = h + 1
        print("Iteration " + str(i) + " Reward: " + str(epReward))