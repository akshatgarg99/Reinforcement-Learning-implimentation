import numpy as np

from gym import Env, spaces
from gym.utils import seeding


def categorical_sampler(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)

    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None
        self.nS = nS
        self.nA = nA
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sampler(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sampler(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transition = self.P[self.s][a]
        i = categorical_sampler([t[0] for t in transition], self.np_random)

        p, s, r, d = transition[i]
        self.s = s
        self.lastaction = a
        return int(s), r, d, {"prob": p}

