# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:51:39 2016

@author: clemens
"""
import numpy as np
import glob
import os
import yaml
import pylab
import matplotlib.pyplot as plt

def load_yamls(directory):
    files = sorted(glob.glob(directory + "/*yml"))
    res = []
    for f in files:
        res.append(yaml.load(open(f, 'r')))
    return res

import pickle

def contextual_bandit(trials, length):
    featurefilename = "hu.npz"
    # load features
    Xy = np.load(featurefilename)
    X = Xy['X']
    y = Xy['y']

    # load trial names
    trialnames = pickle.load(open(featurefilename, 'r'))
    trialinfo = {}
    object_names = []
    pose_names = []
    strategy_names = []
    for t in trialnames:
        print t
        trialinfo[t] = yaml.load(open(t, 'r'))
    
    return None
    
#    res = np.empty(trials, length)
#    for i in xrange(trials):
#        for j in xrange(length):
#            # get random situation
#            situation = np.random.randint(strategies.shape[0])
#            
            

def random_sampling(strategies, trials, length):
    res = np.empty((trials, length))
    for i in range(trials):
        for j in range(length):
            res[i, j] = strategies[np.random.randint(strategies.shape[0]), np.random.randint(strategies.shape[1])]
    return res

def oracle_sampling(strategies, trials, length):
    res = np.empty((trials, length))
    for i in range(trials):
        for j in range(length):
            res[i, j] = np.max(strategies[np.random.randint(strategies.shape[0]),:])
    return res

def greedy_sampling(strategies, trials, length):
    # this is not thompson sampling
    res = np.empty((trials, length))
    
    #np.mean(strategies, axis=0)
    
    for i in xrange(trials):
        reward_theta = np.zeros(strategies.shape[1])
        theta_normalizer = np.zeros(strategies.shape[1])
        
        for j in xrange(length):
            run = np.random.randint(strategies.shape[0])
            mean_rewards = 1. * reward_theta / theta_normalizer
            mean_rewards[np.isnan(mean_rewards)] = 0.5
            draw = np.random.binomial(1, mean_rewards, strategies.shape[1])
            chosen_strategy = np.argmax(draw)

            result = strategies[run, chosen_strategy]
            res[i, j] = result
            
            # update theta
            reward_theta[chosen_strategy] += result
            theta_normalizer[chosen_strategy] += 1
    
    return res

def epsilon_greedy_sampling(strategies, trials, length):
    epsilon = 0.1
    res = np.empty((trials, length))
    
    for i in xrange(trials):
        reward_theta = np.zeros(strategies.shape[1])
        theta_normalizer = np.zeros(strategies.shape[1])
        
        for j in xrange(length):
            run = np.random.randint(strategies.shape[0])
            mean_rewards = 1. * reward_theta / theta_normalizer
            mean_rewards[np.isnan(mean_rewards)] = 0.5
            draw = np.random.binomial(1, mean_rewards, strategies.shape[1])
            chosen_strategy = np.argmax(draw)
            
            if (np.random.rand() < epsilon):
                chosen_strategy = np.random.randint(strategies.shape[1])

            result = strategies[run, chosen_strategy]
            res[i, j] = result
            
            # update theta
            reward_theta[chosen_strategy] += result
            theta_normalizer[chosen_strategy] += 1
    
    return res

def thompson_sampling(strategies, trials, length):
    res = np.empty((trials, length))
    
    for i in range(trials):    
        beta_params = np.ones((2, strategies.shape[1]))
        
        for j in range(length):
            run = np.random.randint(strategies.shape[0])
            
            # draw Bernoulli params
            bernoulli = np.array([np.random.beta(*x) for x in beta_params])
            chosen_strategy = np.argmax(bernoulli)

            result = strategies[run, chosen_strategy]
            res[i, j] = result
            
            # update beta params
            beta_params[chosen_strategy][0] += result
            beta_params[chosen_strategy][1] += (1 - result)
    
    return res

def ucb_sampling(strategies, trials, length):
    res = np.empty((trials, length))
    
    for i in range(trials):        
        reward_theta = np.zeros(strategies.shape[1])
        theta_normalizer = np.zeros(strategies.shape[1])
    
        for j in range(length):
            run = np.random.randint(strategies.shape[0])
            num_pulls = np.max([np.ones(strategies.shape[1]), theta_normalizer], axis=0)
            mean_reward = np.random.binomial(1, 1. * reward_theta / num_pulls, strategies.shape[1])
            chosen_strategy = np.argmax(mean_reward + np.sqrt(np.log(j+1) / num_pulls))
            
            result = strategies[run, chosen_strategy]
            res[i, j] = result
            
            # update theta
            reward_theta[chosen_strategy] += result
            theta_normalizer[chosen_strategy] += 1
        
    return res


def plot_sampling_strategies():
    basedir = "/home/clemens/experimental_data/icra2016/"
    
    # load runs
    all_yamls = [load_yamls(basedir + s) for s in ["surface/chewinggum", "edge/chewinggum", "wall/chewinggum"]]
    runs = np.vstack([[int(y['label']) for y in ys] for ys in all_yamls])
    runs = runs.T
    
    trials = 10000
    length = 50
    
    randomstrat = random_sampling(runs, trials, length)
    oraclestrat = oracle_sampling(runs, trials, length)
    greedystrat = greedy_sampling(runs, trials, length)
    egreedystrat = epsilon_greedy_sampling(runs, trials, length)
    thompstrat = thompson_sampling(runs, trials, length)
    ucbstrat = ucb_sampling(runs, trials, length)
    
    #plt.errorbar(np.arange(50), np.mean(randomstrat, axis=0), np.std(randomstrat, axis=0))
    plt.plot(np.mean(randomstrat, axis=0), label="Random")
    plt.plot(np.mean(greedystrat, axis=0), label="Greedy")
    plt.plot(np.mean(egreedystrat, axis=0), label="Epsilon-Greedy")
    plt.plot(np.mean(ucbstrat, axis=0), label="UCB")
    plt.plot(np.mean(thompstrat, axis=0), label="Thompson")
    plt.plot(np.mean(oraclestrat, axis=0), label="Oracle")
    plt.xlabel("Iteration")
    plt.ylabel("Probability of success")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    
    plt.savefig("sampling_chewinggum.pdf")

def plot_cem():
    all_yamls = [load_yamls(s) for s in ["cem_apple/iter{:02d}".format(i+1) for i in range(10)]]
    runs = np.vstack([[int(y['label']) for y in ys] for ys in all_yamls])
    
    wallforce = np.vstack([[(y['wall_force']) for y in ys] for ys in all_yamls])
    speed = np.vstack([[(y['sliding_speed']) for y in ys] for ys in all_yamls])
    inflation = np.vstack([[(y['finger_inflation']) for y in ys] for ys in all_yamls])
    angle = np.vstack([[(y['angle_of_attack']) for y in ys] for ys in all_yamls])
    positionx = np.vstack([[(y['position_offset_x']) for y in ys] for ys in all_yamls])
    
    plt.bar(np.arange(10)+0.6, np.mean(runs, axis=1))
    plt.ylabel("Probability of success")
    plt.xlabel("Iteration")
    plt.xlim(0.5, 10.5)
    plt.ylim(0, 1.)
    
    plt.savefig("cem_apple_wall.pdf")

def plot_objects():
    basedir = "/home/clemens/experimental_data/icra2016"

    colors = ['r', 'g', 'b']
    strategies = ["surface", "wall", "edge"]
    
    objects = ["apple", "banana", "baseball", "cd", "chewinggum", "creditcard", 
               "egg", "elmer", "expo", "gamepad", "glasses", "kiwi", "marker",
               "mead", "pringles", "screwdriver", "squeak", "tennisball",
               "tissues", "toothpaste", "wallet"]
    
    attempts = np.zeros((len(strategies), len(objects)))
    successes = attempts.copy()
    
    for si, s in enumerate(strategies):
        for oi, o in enumerate(objects):
            try:
                directory = basedir + "/" + s + "/" + o
                if not os.path.isdir(directory):
                    print("{} is missing.".format(directory))
                    pass
                files = glob.glob(directory + "/*yml")
                for f in files:
                    dic = yaml.load(open(f, 'r'))
                    attempts[si, oi] += 1
                    successes[si, oi] += int(dic['label'])
            except Exception as e:
                print e
    
    success_prob = successes / attempts
    
    figure = plt.figure()
    #figure.subplots_adjust(wspace=0,hspace=0.1)
    limits = [-0.75, len(objects) - 0.25, 0, 1.0]
    ind = np.arange(len(objects))
    width = 0.35
    
    colors += ['y']
    strategies += ['max']
    success_prob = np.append(success_prob, np.max(success_prob, axis=0).reshape(1,-1), axis=0)
    
    for si, s in enumerate(strategies):
        ax = figure.add_subplot(len(strategies) * 100 + 11 + si)
        ax.bar(ind - width, success_prob[si], color=colors[si])
        ax.axis(limits)
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(s)
        pylab.xticks(ind, [''] * len(objects))
    plt.xticks(ind, objects, rotation=90)
    
    plt.figtext(0.05, 0.5, 'Success probability', rotation='vertical', ha='left', va='center') 
    figure.subplots_adjust(bottom=0.25)
    
    plt.savefig("objects_success.pdf")
    plt.show()

if __name__ == '__main__':
    #plot_cem()
    plot_objects()
    #plot_sampling_strategies()
    #pass
