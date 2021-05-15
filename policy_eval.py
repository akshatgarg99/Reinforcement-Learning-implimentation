from env import gridworld
import numpy as np

env = gridworld.GridworldEnv()


def policy_eval(policy, env, discount_factor=1, theta=0.00001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + (discount_factor*V[next_state]))
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)



print('running')
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)
print(v)

