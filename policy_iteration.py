from env import gridworld
import numpy as np
from policy_eval import policy_eval


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):

    policy = np.ones([env.nS, env.nA])/env.nA

    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, is_done in env.P[state][a]:
                A[a] += prob*(reward + discount_factor*V[next_state])
        return A

    while True:
        V = policy_eval_fn(policy, env,discount_factor)
        policy_stable = True
        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            action_value = one_step_lookahead(s,V)
            best_a = np.argmax(action_value)

            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        if policy_stable:
            return policy, V


if __name__ == "__main__":
    env = gridworld.GridworldEnv()
    policy, V = policy_improvement(env)
    print(V)
    print(policy)
