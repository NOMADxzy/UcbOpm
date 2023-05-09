import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

cythonized_KLUCB = False
try:
    pass
    # from cklucb import computeKLUCB
except:
    cythonized_KLUCB = False
    pass


class MCTopM(Algorithm):
    def __init__(self, environment, ucb_type="UCB1", policy=0):
        super().__init__(environment=environment)
        self.sensing = True
        self.ucb_type = ucb_type
        # self.klucb = np.vectorize(klucb_ber)
        self.c = 0
        self.step_reward = []
        self.avgacc_reward = []

        self.policy = policy  # 节点（手臂）决策策略：0（本文策略），1（贪心策略），2（always Best Point）

        self.plt_interval = 4  # 绘制点间隔
        self.latency_of_states = [0 for i in range(len(self.env.state))]  # 存放各个state的最终时延，画图用

    def reset(self):
        super().reset()
        self.idx_hist = np.zeros((self.env.M, self.env.K, self.env.T))
        self.s = np.zeros((self.env.M,))
        self.avgacc_reward = []
        self.step_reward = []

    def __str__(self):
        return f"MCTopM"

    def compute_ucb_idx(self, state):
        ucb_idx = np.zeros((self.env.M,
                            self.env.K))

        if self.ucb_type == "UCB1":
            # 取出相应状态下ABCD对每个节点的置信区间上届
            ucb_idx = self.mu_hat[state[0] * self.env.O + state[1]] + np.sqrt(
                (2 * np.log(self.t)) / 1e-10 + self.pulls[state[0] * self.env.O + state[1]])
        else:
            raise ValueError
        if self.policy == 1:  # 贪婪算法，尽量不尝试新节点
            D_pulls = np.sum(self.pulls[:, 3, :], axis=1)
            if len(np.where(D_pulls == 0)[0]) == 0:  # 各个组合都出现过了
                ucb_idx[self.pulls[state[0] * self.env.O + state[1]] == 0] = 0
                return ucb_idx
        ucb_idx[self.pulls[state[0] * self.env.O + state[1]] == 0] = float('inf')
        # ucb_idx = np.minimum(ucb_idx,1)
        return ucb_idx

    def compute_M_hat(self, ucb_idx):  # 用于老虎机决策的矩阵
        # M_hat_t = np.argpartition(-new_ucb_idx, self.env.M-1, axis=1)[:,:self.env.M] # MxM
        arm_idx_sorted = np.argsort(-ucb_idx)
        M_hat = [[] for player in range(self.env.M)]

        for player in range(self.env.M):
            ucb_pivot = ucb_idx[player, arm_idx_sorted[player, self.env.M - 1]]
            for arm in (arm_idx_sorted[player, :]):
                if ucb_idx[player, arm] >= ucb_pivot:
                    M_hat[player].append(arm)
                else:
                    break

        return M_hat

    def update_reward(self, reward_list, regret, step, state_idx):
        reward = np.sum(reward_list)
        self.step_reward.append(reward)
        self.latency_of_states[state_idx] = -reward

        if step == 0:
            self.avgacc_reward.append(reward)
        else:
            self.avgacc_reward.append((step * self.avgacc_reward[-1] + reward) / (step + 1))
        # print("第" + str(self.t) + "轮，reward=" + str(reward) + "，regret=" + str(regret))

    def run(self):
        state_idx = np.random.randint(0, len(self.env.state))
        state = self.env.state[state_idx]
        arms_t = np.random.choice(np.arange(self.env.K),
                                  size=(self.env.M,))
        if self.policy == 2:
            best_idx = self.env.AB_best_point
            arms_t = [best_idx for i in range(self.env.M)]
        # print(arms_t)
        # collisions_t = np.zeros((self.env.M, self.env.K))

        # ucb_idx = self.compute_ucb_idx(state)

        rewards_t, regret_t = self.env.draw(arms_t, state=state, type=2, sensing=self.sensing)
        self.update_reward(rewards_t, regret_t, self.t, state_idx)
        # logger.info(f"drawing {arms_t}, regret:{regret_t}")
        self.update_stats(arms_t=arms_t, state=state, rewards=rewards_t, regret_t=regret_t)
        self.env.update(t=self.t)
        self.t += 1
        # logger.info(f"after update, mu_hat = \n {self.mu_hat}\n")
        while self.t < self.env.T:
            # self.idx_hist[:,:,self.t] = new_ucb_idx # MxK
            # M_hat_t = self.compute_M_hat(ucb_idx=new_ucb_idx)

            new_arms_t = np.zeros((self.env.M,))
            state_idx = np.random.randint(0, len(self.env.state))  # 随机场景和对象
            state = self.env.state[state_idx]

            if self.policy == 2:
                new_arms_t = arms_t
            else:
                new_ucb_idx = self.compute_ucb_idx(state)  # 取出相应场景对象的ucb矩阵
                arm_idx_sorted = np.argsort(-new_ucb_idx)
                for player in range(self.env.M):  # 手臂决策过程

                    best_idx = arm_idx_sorted[player, 0]  # 取ucb值最大的
                    new_arms_t[player] = best_idx

            arms_t = new_arms_t.astype(int)
            # print(arms_t)
            # ucb_idx = new_ucb_idx

            rewards_t, regret_t = self.env.draw(arms_t, state, type=2, sensing=self.sensing)
            self.update_reward(rewards_t, regret_t, self.t, state_idx)

            self.update_stats(arms_t=arms_t, state=state, rewards=rewards_t, regret_t=regret_t)
            self.env.update(t=self.t)
            self.t += 1

    def resize(self, list, a):
        new_list = []
        for idx, e in enumerate(list):
            if idx % a == 0:
                new_list.append(e)
        return new_list


# 下面每个方法绘制一条曲线
    def sample_plot(self, vals, interval, msg):  # 按interval采样vals并绘图
        new_vals = self.resize(vals, interval)
        num_of_x = len(new_vals)
        plt.plot([i * interval for i in range(num_of_x)],
                 new_vals, label=msg)

    def plot_change_K(self):  # 改变节点数量
        msg = str(self.env.K) + "_point"
        self.sample_plot(self.step_reward, self.plt_interval * 2, msg)
        # plt.plot(self.resize(self.step_reward,self.plt_interval * 4),label=msg)
        plt.legend()

    def plot_change_S(self):  # 改变场景数量
        msg = str(self.env.S) + "_scenery"
        self.sample_plot(self.step_reward, self.plt_interval * 3, msg)
        # plt.plot(self.resize(self.step_reward, self.plt_interval * 4), label=msg)
        plt.legend()

    def plot_consume(self):  # 计算资源消耗图
        if self.env.reuse_type == 0:
            msg = "H-MCB"
        elif self.env.reuse_type == 2:
            msg = "Non_reuse"
        else:
            raise ValueError

        # plt.plot(self.resize(self.env.cpu_consume, self.plt_interval), label=msg)
        self.sample_plot(self.env.cpu_consume, self.plt_interval, msg=msg)
        plt.legend()

    def plot_diff_policy(self, reward_type):  # 算法对比图
        msg = ""
        if self.env.reuse_type == 0:
            if self.policy == 0:
                msg = "H-MCB"
            elif self.policy == 0:
                msg = "Greedy"
            elif self.policy == 2:
                msg = "Always-Best"
            else:
                raise NotImplementedError
        elif self.env.reuse_type == 1:
            msg = "MCB"
        elif self.env.reuse_type == 2:
            msg = "Non_reuse"

        # 按间隔绘制折线图，否则太挤
        if reward_type == 'step_reward':
            self.sample_plot(self.step_reward, self.plt_interval, "step_reward_" + msg)
        elif reward_type == 'avgacc_reward':
            self.sample_plot(self.avgacc_reward, self.plt_interval, "avgacc_reward_" + msg)
        else:
            raise ValueError
        plt.legend()


def get_all_algos(config):  # 生成五种算法
    env = Environment(config=config,
                      deterministic=False)
    env_reuse_no_table = Environment(config=config, reuse_type=1,
                                     deterministic=False)
    env_no_reuse = Environment(config=config, reuse_type=2,
                               deterministic=False)
    algos = []
    algos.append(MCTopM(env))  # 本文算法
    algos.append(MCTopM(env_reuse_no_table))  # 无重用指数算法
    algos.append(MCTopM(env_no_reuse))  # 无重用算法
    algos.append(MCTopM(env, policy=1))  # 贪婪算法
    algos.append(MCTopM(env_reuse_no_table, policy=2))  # always BestPoint
    return algos


# 所有的对比图:

def change_K(K_list):  # 本文算法 改变计算节点数量
    plt.figure()
    for K in K_list:
        env = Environment(config={'K': K, 'S': 2},
                          deterministic=False)
        algo = MCTopM(env)
        algo.run()
        algo.plot_change_K()

    plt.xlabel("t")
    plt.ylabel("step_reward")
    if plt_show: plt.show()
    plt.savefig('../results/change_K.png')


def change_S(S_list):  # 本文算法 改变场景数量
    plt.figure()
    for S in S_list:
        env = Environment(config={'K': 100, 'S': S},
                          deterministic=False)
        algo = MCTopM(env)
        algo.run()
        algo.plot_change_S()

    plt.xlabel("t")
    plt.ylabel("step_reward")
    if plt_show: plt.show()
    plt.savefig('../results/change_S.png')


def compare_Latency(val_list, xlabel):  # 五种算法的时延 随节点数量（场景数量）的变化趋势
    plt.figure()
    if xlabel == 'K':
        xlabel_zh = "算力节点数量"
    elif xlabel == 'S':
        xlabel_zh = "场景数量"
    else:
        raise ValueError

    algo_names = ['H-MCB', 'MCB', 'Non_reuse', 'Greedy', 'Always-Best']
    data = {'H-MCB': [],
            'MCB': [],
            'Non_reuse': [],
            'Greedy': [],
            'Always-Best': []}
    for val in val_list:
        algos = get_all_algos(config={xlabel: val})
        for i, algo in enumerate(algos):
            algo.run()
            latency = np.sum(algo.latency_of_states) / len(algo.env.state)
            data[algo_names[i]].append(latency)

    if xlabel == 'S': val_list = [2 * e for e in val_list]
    df = pd.DataFrame(data, index=val_list)
    df.plot(kind='bar')

    # font = fm.FontProperties(fname=r'书法.ttf')
    plt.xlabel(xlabel_zh, fontproperties='simhei')
    plt.ylabel('时延/s', fontproperties='simhei')
    plt.xticks(rotation=360, fontproperties='simhei')
    plt.legend()

    # 显示绘制结果
    if plt_show: plt.show()
    plt.savefig('../results/compare_latency_' + xlabel + '.png')


def change_policy():  # 五种算法 奖励变化情况
    plt.figure()
    config = {'K': 50, 'S': 2,
              'T': 800}
    algos = get_all_algos(config)
    for algo in algos:
        algo.run()
        algo.plot_diff_policy(reward_type='step_reward')

    plt.xlabel("t")
    plt.ylabel("step_reward")
    if plt_show: plt.show()
    plt.savefig('../results/compare_policy_step.png')

    plt.figure()
    for algo in algos:
        algo.run()
        algo.plot_diff_policy(reward_type='avgacc_reward')

    plt.xlabel("t")
    plt.ylabel("avgacc_reward")
    if plt_show: plt.show()
    plt.savefig('../results/compare_policy_avgacc.png')


def compare_Consume():  # 本文算法和无重用算法 对比计算资源消耗
    plt.figure()
    config = {'K': 20, 'S': 2}
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
    if plt_show: plt.show()
    plt.savefig('../results/compare_consume.png')


if __name__ == "__main__":
    if not os.path.exists('../results'):
        os.mkdir('../results')

    plt_show = True

    # change_K([50,100,200])
    # change_S([2,4,6])
    # change_policy()
    # compare_Latency([50,100,200],'K')
    compare_Latency([2, 4, 6], 'S')
    # compare_Consume()
