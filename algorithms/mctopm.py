import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    def __init__(self, environment, ucb_type="UCB1",greedy=False):
        super().__init__(environment=environment)
        self.sensing = True
        self.ucb_type = ucb_type
        # self.klucb = np.vectorize(klucb_ber)
        self.c = 0
        self.step_reward = []
        self.avgacc_reward = []
        self.greedy = greedy
        self.plt_interval = 2
        self.latency_of_states = [0 for i in range(len(self.env.state))] #存放各个state的最终时延，画图用


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
        if self.greedy:#贪婪算法，尽量不尝试新节点
            D_pulls = np.sum(self.pulls[:,3,:],axis=1)
            if len(np.where(D_pulls == 0)[0])==0: #各个组合都出现过了
                ucb_idx[self.pulls[state[0] * self.env.O + state[1]] == 0] = 0
                return ucb_idx
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

    def update_reward(self,reward_list,regret,step,state_idx):
        if step%2 != 0: #图貌似好点
            return
        reward = np.sum(reward_list)
        self.step_reward.append(reward)
        self.latency_of_states[state_idx] = -reward

        if step==0:
            self.avgacc_reward.append(reward)
        else:
            self.avgacc_reward.append((step * self.avgacc_reward[-1] + reward) / (step + 1))
        print("第" + str(self.t) + "轮，reward=" + str(reward) + "，regret=" + str(regret))

    def run(self):
        state_idx = np.random.randint(0,len(self.env.state))
        state = self.env.state[state_idx]
        arms_t = np.random.choice(np.arange(self.env.K),
                                     size=(self.env.M,))
        print(arms_t)
        #collisions_t = np.zeros((self.env.M, self.env.K))

        # ucb_idx = self.compute_ucb_idx(state)

        rewards_t, regret_t = self.env.draw(arms_t,state=state,type=2,sensing=self.sensing)
        self.update_reward(rewards_t,regret_t,self.t,state_idx)
        #logger.info(f"drawing {arms_t}, regret:{regret_t}")
        self.update_stats(arms_t=arms_t, state=state, rewards=rewards_t, regret_t=regret_t)
        self.env.update(t=self.t)
        self.t += 1
        #logger.info(f"after update, mu_hat = \n {self.mu_hat}\n")
        while self.t < self.env.T:
            #self.idx_hist[:,:,self.t] = new_ucb_idx # MxK
            # M_hat_t = self.compute_M_hat(ucb_idx=new_ucb_idx)

            new_arms_t = np.zeros((self.env.M,))
            state_idx = np.random.randint(0, len(self.env.state)) #随机场景和对象
            state = self.env.state[state_idx]
            new_ucb_idx = self.compute_ucb_idx(state) #取出相应场景对象的ucb矩阵

            arm_idx_sorted = np.argsort(-new_ucb_idx)
            for player in range(self.env.M):#手臂决策过程

                best_idx = arm_idx_sorted[player,0]#取ucb值最大的
                new_arms_t[player] = best_idx

            arms_t = new_arms_t.astype(int)
            print(arms_t)
            # ucb_idx = new_ucb_idx

            rewards_t, regret_t  = self.env.draw(arms_t,state,type=2, sensing=self.sensing)
            self.update_reward(rewards_t, regret_t, self.t,state_idx)

            self.update_stats(arms_t=arms_t,state=state, rewards=rewards_t, regret_t=regret_t)
            self.env.update(t=self.t)
            self.t += 1
            #if self.t%1==0:
                ##logger.info(f"Now successes: {self.successes}")
                ##logger.info(f"Now pulls: {self.pulls}")
                ##logger.info(f"Now collisions: {self.collisions}")
                #logger.info(f"t = {self.t}: drawing {arms_t}, reward: {rewards_t}, regret:{regret_t}")
                #logger.info(f"Now mu_hat: \n{self.mu_hat}\n, successes: \n {self.successes}, pulls:\n{self.pulls}\n")


    def resize(self,list,a):
        new_list = []
        for idx,e in enumerate(list):
            if idx % a ==0:
                new_list.append(e)
        return new_list

    def plot_change_K(self):
        msg = str(self.env.K)+"_point"
        plt.plot(self.resize(self.step_reward,self.plt_interval * 4),label=msg)
        plt.legend()

    def plot_change_S(self):
        msg = str(self.env.S) + "_scenery"
        plt.plot(self.resize(self.step_reward, self.plt_interval * 4), label=msg)
        plt.legend()

    def plot_consume(self):
        if self.env.reuse_type==0:
            msg = "reuse"
        elif self.env.reuse_type==2:
            msg = "no_reuse"
        else: raise ValueError

        plt.plot(self.resize(self.env.cpu_consume, self.plt_interval), label=msg)
        plt.legend()

    def plot_diff_policy(self):
        msg = ""
        if self.env.reuse_type == 0:
            if self.greedy:
                msg = "greedy"
            else: msg = "reuse_with_reuse_table"
        elif self.env.reuse_type == 1:
            msg = "reuse_without_reuse_table"
        elif self.env.reuse_type == 2:
            msg = "no_reuse"
        plt.plot(self.resize(self.step_reward,self.plt_interval), label="step_reward_" + msg)
        plt.plot(self.resize(self.avgacc_reward,self.plt_interval), label="avgacc_reward_" + msg)
        plt.legend()

def get_all_algos(config):
    env = Environment(config=config,
                      deterministic=False)
    env_reuse_no_table = Environment(config=config, reuse_type=1,
                                     deterministic=False)
    env_no_reuse = Environment(config=config, reuse_type=2,
                               deterministic=False)
    algos = []
    algos.append(MCTopM(env)) #本文算法
    algos.append(MCTopM(env_reuse_no_table)) #无重用指数算法
    algos.append(MCTopM(env_no_reuse)) #无重用算法
    algos.append(MCTopM(env, greedy=True)) #贪婪算法
    return algos

def change_K(K_list):
    for K in K_list:
        env = Environment(config={'K': K, 'S': 2},
                          deterministic=False)
        algo = MCTopM(env)
        algo.run()
        algo.plot_change_K()

    plt.xlabel("t")
    plt.ylabel("step_reward")
    plt.show()



def change_S(S_list):
    for S in S_list:
        env = Environment(config={'K': 100, 'S': S},
                          deterministic=False)
        algo = MCTopM(env)
        algo.run()
        algo.plot_change_S()

    plt.xlabel("t")
    plt.ylabel("step_reward")
    plt.show()

def compare_Latency(val_list,xlabel):
    if xlabel=='K': xlabel_zh = "节点数量"
    elif xlabel=='S': xlabel_zh = "场景数量"
    else: raise ValueError

    algo_names = ['本文算法','无重用指数','无重用','贪婪算法']
    data = {'本文算法': [],
            '无重用指数': [],
            '无重用': [],
            '贪婪算法': []}
    for val in val_list:
        algos = get_all_algos(config={xlabel: val})
        for i, algo in enumerate(algos):
            algo.run()
            latency = np.sum(algo.latency_of_states) / len(algo.env.state)
            data[algo_names[i]].append(latency)

    df = pd.DataFrame(data, index=val_list)
    df.plot(kind='bar')


    # font = fm.FontProperties(fname=r'书法.ttf')
    plt.xlabel(xlabel_zh, fontproperties='simhei')
    plt.ylabel('时延', fontproperties='simhei')
    plt.xticks(rotation=90, fontproperties='simhei')
    plt.legend()

    # 显示绘制结果
    plt.show()

def change_policy():
    config = {'K': 100, 'S': 2}
    algos = get_all_algos(config)
    for algo in algos:
        algo.run()
        algo.plot_diff_policy()

    plt.xlabel("t")
    plt.ylabel("reward")
    plt.show()

def compare_Consume():
    config = {'K': 100, 'S': 2}
    env = Environment(config=config,
                      deterministic=False)

    env_no_reuse = Environment(config=config, reuse_type=2,
                               deterministic=False)
    algos = []
    algos.append(MCTopM(env))
    algos.append(MCTopM(env_no_reuse))
    for algo in algos:
        algo.run()
        algo.plot_consume()

    plt.xlabel("t")
    plt.ylabel("consume")
    plt.show()



if __name__ == "__main__":
    # change_K([50,100,200])
    # change_S([2,4,6])
    # change_policy()
    # compare_Latency([50,100,200],'K')
    # compare_Latency([2, 4, 6], 'S')
    compare_Consume()
