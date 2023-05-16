import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import random
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, config,reuse_type=0, deterministic=False):
        if not 'K' in config: config['K'] = 100
        if not 'S' in config: config['S'] = 2 #默认值

        self.K = config['K'] #手臂数量
        self.M = 4 #服务数量

        self.S = config['S'] #(scenery)场景数量
        self.O = 2 #(object)对象数量

        if not 'T' in config: config['T'] = self.T = self.K * self.S * self.O * 2  # 这个比例时画图较好
        self.T = config['T'] #这个比例时画图较好
        # 计算参数
        self.phi = 0.5
        self.tao = 0.018  # 检索时延

        self.state = [] #枚举各种场景对象组合
        for s in range(self.S):
            for o in range(self.O):
                st = (s, o)
                self.state.append(st)

        #使用现有ability数据
        with open('ability.pkl',"rb") as file:
            ability = pickle.load(file)
        self.cpu_ability = ability[0][:self.K]
        self.net_ability = ability[1][:self.K]
        #重新生成ability
        # self.cpu_ability = self.get_random_gauss_list(1, 10, self.K)
        # self.net_ability = self.get_random_gauss_list(1, 10, self.K)
        # with open('ability.pkl','wb') as file:
        #     file.write(pickle.dumps([self.cpu_ability,self.net_ability]))

        self.best_Ks = np.argsort(-(np.asarray(self.cpu_ability) + np.asarray(self.net_ability)))[
                       :2 * (self.S + self.S * self.O)]
        self.AB_best_point = self.get_AB_best_point()  # 对AB服务最好的节点，AB后期会一直使用此节点
        self.Cache = [[], []]  # 第一处放场景缓存、第二处放对象缓存
        i = 0
        for s in range(self.S):
            pair = list(self.best_Ks[i:i + 2])
            if self.net_ability[pair[0]] < self.net_ability[pair[1]]:
                pair = [pair[1], pair[0]]  # 带宽强的排前面，方便计算regret
            self.Cache[0].append(pair)
            i = i + 2
            for o in range(self.O):
                pair = list(self.best_Ks[i:i + 2])
                if self.net_ability[pair[0]] < self.net_ability[pair[1]]:
                    pair = [pair[1], pair[0]]  # 带宽强的排前面
                self.Cache[1].append(pair)
                i = i + 2

        self.reuse_table_h = np.ones((2+self.S + self.S * self.O, self.K)) #2存放A和B，self.S存放不同场景，self.S*self.O存放不同对象
        self.reuse_table_z = np.zeros((2+self.S + self.S * self.O, self.K))
        self.Z = 5  # 已有连续Z次未执行重用缓存检索，则下次必定执行重用缓存检索
        self.reuse_type = reuse_type #有重用指数重用，无重用指数重用，无重用，贪婪算法

        self.cpu_consume = []

        # self.mu = config['mu']
        # self.mu = np.array(sorted(self.mu, reverse=True))
        # self.mu_opt_M = np.sort(self.mu)[-self.M:]
        # self.mu_opt = np.sum(self.mu[:self.M])
        # self.K = len(self.mu)
        # logger.info(
        #     f"Created environment with M = {self.M}, mu = {self.mu}, mu_opt = {self.mu_opt}, T = {self.T}, K = {self.K}")
        self.dynamic = False
        self.deterministic = deterministic

    def __str__(self):
        return f"M{self.M}-K{self.K}-mu{str(self.mu)}"

    def get_random_gauss_list(self,bottom,top,size):
        val = 0
        val_list = []
        while len(val_list)<size:
            val = random.gauss(5, 4)
            while val<bottom or val>top:
                val = random.gauss(5,4)
            val_list.append(val)
        return val_list

    def get_AB_best_point(self): #对AB服务来说最好的节点（计算能力和网络能力综合最强）
        val, idx = 10000, -1
        for i in range(self.K):
            new_val = 1 / self.net_ability[i] + 1 / self.cpu_ability[i]
            if new_val < val:
                val = new_val
                idx = i
        return idx

    def determine_reuse(self, idx, point):
        if self.reuse_table_z[idx, point] >= self.Z:
            self.reuse_table_z[idx, point] = 0
            return True
        return self.reuse_table_h[idx, point] == 1

    def update_reuse_table(self, idx, point, reuse_result):
        self.reuse_table_h[idx, point] = reuse_result
        if reuse_result == 1:
            self.reuse_table_z[idx, point] = 0
        else:
            self.reuse_table_z[idx, point] += 1

    def update(self, t):
        pass

    def nearby(self,ability_expect,delta=0.2): #delta表示上（下）波动的范围
        ability = np.random.normal(ability_expect, delta/3, 1)
        return ability

    def draw(self, arms, state, type=0, sensing=False):
        if self.reuse_type == 1:
            return self.draw_without_reuse_table(arms, state)
        elif self.reuse_type == 2:
            return self.draw_without_reuse(arms, state)

        rewards = np.zeros((self.M,))  # 每个子任务的奖励
        regret_t = 0  # 与最优情况奖励的差值
        num_cpt = 0  # 本轮调用计算的次数
        for player in range(self.M):
            point = arms[player]

            rewards[player] = -(self.phi / self.nearby(self.net_ability[point]))  # 传输时延
            if player < 2:
                reuse = self.determine_reuse(player, point)
                if reuse:
                    rewards[player] += -self.tao
                    self.update_reuse_table(player, point, reuse_result=-1)
                else: self.update_reuse_table(player, point, reuse_result=0)
                rewards[player] += -( self.phi / self.nearby(self.cpu_ability[point]))
                regret_t += -(self.phi / self.net_ability[self.AB_best_point] +
                            self.tao + self.phi / self.cpu_ability[self.AB_best_point])
                # regret_t += rewards[player]
                num_cpt += 1
            elif player == 2:  # C服务
                reuse = self.determine_reuse(2+state[0], point)
                if not reuse:
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                    num_cpt += 1
                    self.update_reuse_table(2+state[0], point, reuse_result=0)
                else:
                    rewards[player] += -self.tao
                    if point in self.Cache[0][state[0]]:  # 节点上有缓存
                        self.update_reuse_table(2+state[0], point, reuse_result=1)
                    else:
                        rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                        num_cpt += 1
                        self.update_reuse_table(2+state[0], point, reuse_result=-1)
                regret_t += -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)
                # if -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)<rewards[player]:
                #     print(-(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao))
                #     print(rewards[player])
            else:  # D服务
                idx = self.S + state[0] * self.O + state[1]  # 表的前S行是C服务的缓存
                reuse = self.determine_reuse(2+idx, point)
                if not reuse:
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                    num_cpt += 1
                    self.update_reuse_table(2+idx, point, reuse_result=0)
                else:
                    rewards[player] += -self.tao
                    if point in self.Cache[1][state[0] * self.O + state[1]]:  # 检索命中
                        self.update_reuse_table(2+idx, point, reuse_result=1)
                    else:  # 检索未命中
                        rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                        num_cpt += 1
                        self.update_reuse_table(2+idx, point, reuse_result=-1)
                regret_t += -(self.phi / self.net_ability[self.Cache[1][state[0] * self.O + state[1]][0]] + self.tao)
                # if -(self.phi / self.net_ability[self.Cache[1][state[0]*self.O+state[1]][0]] + self.tao)<rewards[player]:
                #     print(-(self.phi / self.net_ability[self.Cache[1][state[0]*self.O+state[1]][0]] + self.tao))
                #     print(rewards[player])

            regret_t -= rewards[player]
        self.cpu_consume.append(num_cpt*self.phi)
        return rewards, regret_t

    def draw_without_reuse(self, arms, state):
        # counts = collections.Counter(arms)
        rewards = np.zeros((self.M,))
        regret_t = 0
        for player in range(self.M):
            point = arms[player]
            rewards[player] = -self.phi / self.nearby(self.net_ability[point])  # 传输时延
            rewards[player] += -( self.phi / self.nearby(self.cpu_ability[point]))  # 计算时延
            regret_t += -(self.phi / self.net_ability[self.AB_best_point]
                          + self.phi / self.cpu_ability[self.AB_best_point])  # 最优奖励
            # regret_t += rewards[player]
            regret_t -= rewards[player]  # 减去实际奖励 即得到 regret
        self.cpu_consume.append(4 * self.phi)  # 必有4次计算
        return rewards, regret_t

    def draw_without_reuse_table(self, arms, state):
        rewards = np.zeros((self.M,))
        regret_t = 0
        num_cpt = 0
        for player in range(self.M):
            point = arms[player]
            rewards[player] = -self.phi / self.nearby(self.net_ability[point])  # 传输时延
            rewards[player] += -self.tao #每次都要检索
            if player < 2:
                rewards[player] +=  -( self.phi / self.nearby(self.cpu_ability[point])) #计算时延
                regret_t += -(self.phi / self.net_ability[self.AB_best_point] + (
                            self.tao + self.phi / self.cpu_ability[self.AB_best_point]))
                # regret_t += rewards[player]
                num_cpt += 1
            elif player == 2:  # C服务
                if point not in self.Cache[0][state[0]]:  # 节点上没有缓存
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                    num_cpt += 1
                regret_t += -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)
                # if -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)<rewards[player]:
                #     print(-(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao))
                #     print(rewards[player])
            else:  # D服务
                if point not in self.Cache[1][state[0] * self.O + state[1]]:  # 检索未命中
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                    num_cpt += 1
                regret_t += -(self.phi / self.net_ability[self.Cache[1][state[0] * self.O + state[1]][0]] + self.tao)
                # if -(self.phi / self.net_ability[self.Cache[1][state[0]*self.O+state[1]][0]] + self.tao)<rewards[player]:
                #     print(-(self.phi / self.net_ability[self.Cache[1][state[0]*self.O+state[1]][0]] + self.tao))
                #     print(rewards[player])

            regret_t -= rewards[player]

        self.cpu_consume.append(num_cpt * (self.phi + random.uniform(0, 0.1)))
        return rewards, regret_t

    def draw_greedy(self):
        return 0

