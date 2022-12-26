import numpy as np
import matplotlib.pyplot as plt

import collections
import os

import logging
import pickle

import time
logger = logging.getLogger(__name__)
from tqdm import tqdm


class Algorithm:
    def __init__(self, environment, save_optional=False):
        self.env = environment
        self.sensing = False
        self.collision_sensing = False

        self.save_optional = save_optional
        self.reset()

    def reset(self):
        self.t = 0
        self.regret = np.zeros(self.env.T)
        self.pulls = np.zeros((self.env.S*self.env.O,self.env.M,
                               self.env.K),  dtype=np.int32)
        self.successes = np.zeros((self.env.S*self.env.O,self.env.M,
                                   self.env.K))
        self.mu_hat = np.zeros((self.env.S*self.env.O,self.env.M,
                               self.env.K))


        self.collisions = np.zeros((self.env.M, self.env.K)) # self.collisions[i,j] = nb of collisions of player i on arm j

        self.collision_hist = np.zeros((self.env.K, self.env.T))

        self.delta_hist = []
        self.success_hist = []
        self.pulls_hist = []

        if self.env.dynamic:
            self.sum_mu_opt = 0
            self.system_reward_tot = 0


        if self.save_optional:
            self.arm_history = np.zeros((self.env.M, self.env.T))
        if self.env.K == 2 and self.save_optional:
            self.delta_hist = np.zeros((self.env.M, self.env.T))
            self.success_hist = np.zeros((self.env.M,self.env.K, self.env.T))
            self.pulls_hist = np.zeros((self.env.M, self.env.K, self.env.T))

    def arms_t_policy(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
        self.reset()
        while self.t < self.env.T:
            arms_t = self.arms_t_policy() #arms_t，列表，第i个元素表示用户i推了arms_t[i]号杆
            rewards_t, regret_t = self.env.draw(arms_t,)

            self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t)
            self.env.update(t=self.t)
            self.t += 1


    def compute_mu_hat(self):
        mu_hat = self.successes / (1e-7+self.pulls)
        mu_hat[self.pulls == 0] = 0 # if arm has never been pulled, mu_hat =0
        return mu_hat

    def update_stats(self, arms_t, state, rewards, regret_t, system_reward=0):
        """
        arms_t: vector of size env.M (number of players currently in the game)
        """
        if self.t < self.env.T:
            self.regret[self.t] = regret_t
            self.pulls[:, 0, arms_t[0]] += 1 #A服务
            self.pulls[:, 1, arms_t[1]] += 1 #B服务
            self.pulls[state[0]*self.env.O:(state[0]+1)*self.env.O,2,arms_t[2]] += 1 #C服务
            self.pulls[state[0]*self.env.O+state[1],3,arms_t[3]] += 1 #D服务

            assert (np.sum(self.pulls[0,0]) == self.t+1)
            assert (np.sum(self.pulls[:,3]) == self.t+1) #检验

            #同上
            self.successes[:, 0, arms_t[0]] += rewards[0]
            self.successes[:, 1, arms_t[1]] += rewards[1]
            self.successes[state[0] * self.env.O:(state[0] + 1) * self.env.O, 2, arms_t[2]] += rewards[2]
            self.successes[state[0] * self.env.O + state[1], 3, arms_t[3]] += rewards[3]
            self.mu_hat = self.compute_mu_hat()

            # self.collisions += collisions_t
            # self.collision_hist[:,self.t] += np.max(collisions_t, axis=0)



    def reset_player(self, leaving_players):
        """
        (dynamic setting), when a player leaves, its statistics are resetted
        """
        for idx_player in leaving_players:
            self.mu_hat[:,idx_player,:] = 0
            self.pulls[:,idx_player,:] = 0
            self.successes[:,idx_player,:] = 0




    def plot_arm_history(self,
                         min_T=0,
                         max_T=None,
                        ):
        if max_T is None:
            max_T = self.env.T
        #fig = plt.figure(figsize=(12,2))
        #fig.add_subplot(111)
        for player in range(self.env.M):
            player_arm_history = list(self.arm_history[player,:])
            plt.step(np.arange(min_T,max_T),
                     player_arm_history[min_T:max_T],
                     where="post",
                     label=f"user {player}")

        plt.ylabel("arms")
        plt.xlabel("t")
        plt.legend(loc="lower right")


    def plot_delta_history(self,
                         min_T=0,
                         max_T=None,
                         y_min=-1,
                         y_max=1
                        ):
        if max_T is None:
            max_T = self.env.T

        name = f"delta"# T_{min_T}_{max_T}"


        for player in range(self.env.M):
            player_hist = self.delta_hist[player,:]
            plt.step(np.arange(min_T,max_T),
                     player_hist[min_T:max_T],
                     where="post",
                     label=f"user {player} ")
        plt.plot(np.arange(min_T,max_T),
                 np.zeros((max_T-min_T,)))
        plt.ylabel(name)
        plt.xlabel("t")
        plt.legend(loc="lower right")
        plt.title(name)
        plt.grid()
        plt.ylim(y_min,y_max)


    def plot_mu_hat(self,min_T,max_T):
        if max_T is None:
            max_T = self.env.T
        mu_hat_hist = self.success_hist/self.pulls_hist
        mu_hat_hist[self.pulls_hist == 0] = 0
        for arm in range(self.env.K):
            plt.plot(np.arange(min_T,max_T),
                     self.env.mu[arm] * np.ones((max_T-min_T,)),"--"
                    )
            for player in range(self.env.M):
                plt.step(np.arange(min_T,max_T),mu_hat_hist[player,arm,min_T:max_T],label=f"user {player}", where="post",)
        plt.plot(np.arange(min_T,max_T),
                 np.zeros((max_T-min_T,)),
                )
        plt.ylabel("mu_hat")
        plt.xlabel("t")
        plt.legend(loc="lower right")
        plt.grid()
