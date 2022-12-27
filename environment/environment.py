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

        self.T = self.K*self.S*self.O*2 #这个比例时画图较好
        # 计算参数
        self.phi = 0.5
        self.tao = 0.05

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

        self.reuse_table_h = np.ones((self.S + self.S * self.O, self.K))
        self.reuse_table_z = np.zeros((self.S + self.S * self.O, self.K))
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

        rewards = np.zeros((self.M,))
        regret_t = 0
        num_cpt = 0
        for player in range(self.M):
            point = arms[player]

            rewards[player] = -(self.phi / self.nearby(self.net_ability[point]))  # 传输时延
            if player < 2:
                rewards[player] += -(self.tao + self.phi / self.nearby(self.cpu_ability[point]))
                regret_t += -(self.phi / self.net_ability[self.AB_best_point] + (
                            self.tao + self.phi / self.cpu_ability[self.AB_best_point]))
                # regret_t += rewards[player]
                num_cpt += 1
            elif player == 2:  # C服务
                reuse = self.determine_reuse(state[0], point)
                if not reuse:
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                    num_cpt += 1
                    self.update_reuse_table(state[0], point, reuse_result=0)
                else:
                    rewards[player] += -self.tao
                    if point in self.Cache[0][state[0]]:  # 节点上有缓存
                        self.update_reuse_table(state[0], point, reuse_result=1)
                    else:
                        rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                        num_cpt += 1
                        self.update_reuse_table(state[0], point, reuse_result=-1)
                regret_t += -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)
                # if -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)<rewards[player]:
                #     print(-(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao))
                #     print(rewards[player])
            else:  # D服务
                idx = self.S + state[0] * self.O + state[1]  # 表的前S行是C服务的缓存
                reuse = self.determine_reuse(idx, point)
                if not reuse:
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                    num_cpt += 1
                    self.update_reuse_table(idx, point, reuse_result=0)
                else:
                    rewards[player] += -self.tao
                    if point in self.Cache[1][state[0] * self.O + state[1]]:  # 检索命中
                        self.update_reuse_table(idx, point, reuse_result=1)
                    else:  # 检索未命中
                        rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                        num_cpt += 1
                        self.update_reuse_table(idx, point, reuse_result=-1)
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
            rewards[player] += (-(self.tao + self.phi / self.nearby(self.cpu_ability[point])))
            regret_t += -(self.phi / self.net_ability[self.AB_best_point] + (
                    self.tao + self.phi / self.cpu_ability[self.AB_best_point]))
            # regret_t += rewards[player]
            regret_t -= rewards[player]
        self.cpu_consume.append(4 * self.phi)
        return rewards, regret_t

    def draw_without_reuse_table(self, arms, state):
        rewards = np.zeros((self.M,))
        regret_t = 0
        for player in range(self.M):
            point = arms[player]
            rewards[player] = -self.phi / self.nearby(self.net_ability[point])  # 传输时延
            if player < 2:
                rewards[player] +=  -(self.tao + self.phi / self.nearby(self.cpu_ability[point]))
                regret_t += -(self.phi / self.net_ability[self.AB_best_point] + (
                            self.tao + self.phi / self.cpu_ability[self.AB_best_point]))
                # regret_t += rewards[player]
            elif player == 2:  # C服务
                rewards[player] += -self.tao
                if point not in self.Cache[0][state[0]]:  # 节点上没有缓存
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                regret_t += -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)
                # if -(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao)<rewards[player]:
                #     print(-(self.phi / self.net_ability[self.Cache[0][state[0]][0]] + self.tao))
                #     print(rewards[player])
            else:  # D服务
                rewards[player] += -self.tao
                if point not in self.Cache[1][state[0] * self.O + state[1]]:  # 检索未命中
                    rewards[player] += -self.phi / self.nearby(self.cpu_ability[point])
                regret_t += -(self.phi / self.net_ability[self.Cache[1][state[0] * self.O + state[1]][0]] + self.tao)
                # if -(self.phi / self.net_ability[self.Cache[1][state[0]*self.O+state[1]][0]] + self.tao)<rewards[player]:
                #     print(-(self.phi / self.net_ability[self.Cache[1][state[0]*self.O+state[1]][0]] + self.tao))
                #     print(rewards[player])

            regret_t -= rewards[player]

        return rewards, regret_t

    def draw_greedy(self):
        return 0


class DynamicEnvironment(Environment):
    def __init__(self, config, players_can_leave=False, deterministic=False,
                 t_entries=None,
                 t_leaving=None,
                 lambda_poisson=1 / 10000,
                 mu_exponential=10000):
        super().__init__(config=config, deterministic=deterministic)
        assert self.M == self.K
        self.dynamic = True
        self.players_can_leave = players_can_leave

        self.lambda_poisson = lambda_poisson
        self.mu_exponential = mu_exponential  # how many steps do they stay? ~10 seconds  = 10k steps

        if t_entries:
            self.t_entries = t_entries
        else:
            self.sample_t_entries()
        self.t_entries.sort()
        self.t_entries[0] = 0
        print("Nb entrées:", len(self.t_entries))
        assert self.t_entries[1] != 0

        if players_can_leave:
            if t_leaving:
                self.t_leaving = t_leaving
            else:
                self.sample_t_leaving()
            if not (self.t_leaving > self.t_entries).all():
                self.t_leaving[np.argwhere(self.t_leaving == self.t_entries).flatten()] += 1

            assert (
                    self.t_leaving > self.t_entries).all(), f"{self.t_entries[np.argwhere(self.t_leaving <= self.t_entries).flatten()]}, sorties:{self.t_leaving[np.argwhere(self.t_leaving <= self.t_entries).flatten()]}"
            assert len(np.unique(self.t_leaving) == len(self.t_leaving))
            self.t_leaving_dict = dict(zip(self.t_leaving, [None for i in range(len(self.t_leaving))]))
            self.t_leaving_dict[self.t_leaving[0]] = 0
        else:
            self.t_leaving = []  # self.T*np.ones(self.M)
            self.t_leaving_dict = {}
            self.individual_horizons = np.zeros((self.M,))
            for player in range(self.M):
                self.individual_horizons[player] = self.T - self.t_entries[
                    player]  # il n'y a que les M premiers qui peuvent entrer dans le game

        self.active_players = {0}
        self.inactive_players = {i for i in range(1, self.M)}

        self.mu_opt = np.sum(self.mu[:len(self.active_players)])
        print("mu_optinit:", self.mu_opt)
        self.ith_entry = 1

    def sample_t_entries(self):
        n_entries = max(self.M, np.random.poisson(self.lambda_poisson * self.T))

        self.t_entries = np.random.randint(self.T, size=n_entries)

    def sample_t_leaving(self):
        staying_time = np.random.exponential(self.mu_exponential, size=len(self.t_entries))
        self.t_leaving = np.minimum(self.t_entries + (staying_time).astype(int), self.T)

    def update(self, t):
        '''
        Leaving players have played the round t
        '''
        env_has_changed = False

        leaving_players = []

        while t in self.t_leaving_dict:
            idx_player = self.t_leaving_dict[t]
            leaving_players.append(idx_player)
            self.active_players.remove(idx_player)
            self.inactive_players.add(idx_player)
            # print(f"........t = {t}: player of idx {idx_player} leaves the game")
            del self.t_leaving_dict[t]
            env_has_changed = True

        while self.ith_entry < len(self.t_entries) and t == self.t_entries[self.ith_entry]:
            if len(self.active_players) == self.M:
                # print(f"__t = {t}: saturated system, player at entry {t} cannot enter")
                if not self.players_can_leave:
                    return leaving_players
                if self.t_leaving[self.ith_entry] in self.t_leaving_dict:
                    del self.t_leaving_dict[self.t_leaving[self.ith_entry]]
                self.ith_entry += 1

            for idx_player in range(self.M):
                if idx_player not in self.active_players:
                    self.active_players.add(idx_player)
                    self.inactive_players.remove(idx_player)
                    if self.players_can_leave:
                        self.t_leaving_dict[self.t_leaving[self.ith_entry]] = idx_player
                    # print(f"........t = {t}: player {self.ith_entry} enters the game at player slot {idx_player}")
                    self.ith_entry += 1
                    break
            env_has_changed = True

        if env_has_changed:
            self.mu_opt = np.sum(self.mu[:len(self.active_players)])
            # print(f"........t = {t}: active players: {self.active_players} ({len(self.active_players)}). new mu_opt = {self.mu_opt}")

        return leaving_players

    def draw(self, arms):
        arms[list(self.inactive_players)] = -1
        counts = collections.Counter(arms)
        rewards = np.zeros((self.M,))
        regret_t = self.mu_opt
        collisions_t = np.zeros((self.M, self.K))
        for player in self.active_players:
            if counts[arms[player]] >= 2:  # There was collision
                rewards[player] = 0
                collisions_t[player, arms[player]] = 1
            else:
                if self.deterministic:
                    rewards[player] = self.mu[arms[player]]
                else:
                    rewards[player] = np.random.binomial(n=1, p=self.mu[arms[player]])
                regret_t -= self.mu[arms[player]]  # rewards[player]

        system_reward = self.mu_opt - regret_t
        # print("draw, ", arms, collisions_t)
        return rewards, regret_t, collisions_t, system_reward
