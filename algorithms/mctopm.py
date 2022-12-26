import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from tqdm import tqdm
# from .algorithm import *
# from .utils import *
from algorithm import Algorithm
from environment.environment import Environment

logger = logging.getLogger()
logger.setLevel(logging.INFO)


cythonized_KLUCB = True
try:
    from cklucb import computeKLUCB
except:
    cythonized_KLUCB = False
    pass

class MCTopM(Algorithm):
    def __init__(self, environment, ucb_type="UCB1"):
        super().__init__(environment=environment)
        self.sensing = True
        self.ucb_type = ucb_type
        # self.klucb = np.vectorize(klucb_ber)
        self.c = 0
        self.step_reward = []
        self.avgacc_reward = []


    def reset(self):
        super().reset()
        self.idx_hist = np.zeros((self.env.M, self.env.K, self.env.T))
        self.s = np.zeros((self.env.M,))
        self.avgacc_reward = []
        self.step_reward = []
    def __str__(self):
        return f"MCTopM"

    def compute_ucb_idx(self,state):
        ucb_idx = np.zeros((self.env.M,
                            self.env.K))
        if self.ucb_type == "UCB1":
            #取出相应状态下ABCD对每个节点的置信区间上届
            ucb_idx = self.mu_hat[state[0]*self.env.O+state[1]] + np.sqrt((2*np.log(self.t))/1e-10+self.pulls[state[0]*self.env.O+state[1]])
        else:
            raise ValueError
        ucb_idx[self.pulls[state[0]*self.env.O+state[1]]== 0] = float('inf')
       # ucb_idx = np.minimum(ucb_idx,1)
        return ucb_idx

    def compute_M_hat(self, ucb_idx):
        #M_hat_t = np.argpartition(-new_ucb_idx, self.env.M-1, axis=1)[:,:self.env.M] # MxM
        arm_idx_sorted = np.argsort(-ucb_idx)
        M_hat = [[] for player in range(self.env.M)]

        for player in range(self.env.M):
            ucb_pivot = ucb_idx[player, arm_idx_sorted[player,self.env.M-1]]
            for arm in (arm_idx_sorted[player,:]):
                if ucb_idx[player, arm] >= ucb_pivot:
                    M_hat[player].append(arm)
                else:
                    break

        return M_hat

    def update_reward(self,reward_list,regret,step):
        if step%5 != 0:
            return
        reward = np.sum(reward_list)/4
        self.step_reward.append(reward)
        if step==0:
            self.avgacc_reward.append(reward)
        else:
            self.avgacc_reward.append((step * self.avgacc_reward[-1] + reward) / (step + 1))
        print("第" + str(self.t) + "轮，reward=" + str(reward) + "，regret=" + str(regret))

    def run(self):
        state = self.env.state[np.random.randint(0,4)]
        arms_t = np.random.choice(np.arange(self.env.K),
                                     size=(self.env.M,))
        #collisions_t = np.zeros((self.env.M, self.env.K))

        ucb_idx = self.compute_ucb_idx(state)

        rewards_t, regret_t = self.env.draw(arms_t,state=state,type=2,sensing=self.sensing)
        self.update_reward(rewards_t,regret_t,self.t)
        #logger.info(f"drawing {arms_t}, regret:{regret_t}")
        self.update_stats(arms_t=arms_t, state=state, rewards=rewards_t, regret_t=regret_t)
        self.env.update(t=self.t)
        self.t += 1
        #logger.info(f"after update, mu_hat = \n {self.mu_hat}\n")
        while self.t < self.env.T:
            #self.idx_hist[:,:,self.t] = new_ucb_idx # MxK
            # M_hat_t = self.compute_M_hat(ucb_idx=new_ucb_idx)

            new_arms_t = np.zeros((self.env.M,))
            state = self.env.state[np.random.randint(0,4)]
            new_ucb_idx = self.compute_ucb_idx(state)

            for player in range(self.env.M):#手臂决策过程
                arm_idx_sorted = np.argsort(-new_ucb_idx)
                best_idx = arm_idx_sorted[player,0]#取ucb值最大的
                new_arms_t[player] = best_idx

            arms_t = new_arms_t.astype(int)
            print(arms_t)
            ucb_idx = new_ucb_idx

            rewards_t, regret_t  = self.env.draw(arms_t,state,type=2, sensing=self.sensing)
            self.update_reward(rewards_t, regret_t, self.t)

            self.update_stats(arms_t=arms_t,state=state, rewards=rewards_t, regret_t=regret_t)
            self.env.update(t=self.t)
            self.t += 1
            #if self.t%1==0:
                ##logger.info(f"Now successes: {self.successes}")
                ##logger.info(f"Now pulls: {self.pulls}")
                ##logger.info(f"Now collisions: {self.collisions}")
                #logger.info(f"t = {self.t}: drawing {arms_t}, reward: {rewards_t}, regret:{regret_t}")
                #logger.info(f"Now mu_hat: \n{self.mu_hat}\n, successes: \n {self.successes}, pulls:\n{self.pulls}\n")

        msg = ""
        if self.env.reuse_type==0:
            msg = "reuse_with_table"
        elif self.env.reuse_type==1:
            msg = "no_reuse"
        else:
            msg = "reuse_without_reuse_table"
        plt.plot(self.step_reward,label="step_reward_"+msg)
        plt.plot(self.avgacc_reward,label="avgacc_reward_"+msg)
        plt.legend()

if __name__ == "__main__":
    config = {'T':800,'dynamic':False}
    env = Environment(config=config,
                      deterministic=False)
    env1 = Environment(config=config,reuse_type=1,
                      deterministic=False)
    env2 = Environment(config=config,reuse_type=2,
                      deterministic=False)
    algo = MCTopM(env)
    algo1 = MCTopM(env1)
    algo2 = MCTopM(env2)

    algo.run()
    algo1.run()
    algo2.run()

    plt.xlabel("step")
    plt.ylabel("reward")
    plt.show()