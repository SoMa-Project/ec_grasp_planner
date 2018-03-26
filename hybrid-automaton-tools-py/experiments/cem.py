#!/usr/bin/env python
import rospy
import roslib
import actionlib
import numpy as np
import subprocess
import os
import signal
import time
import sys
import argparse
import math
import yaml
import datetime
import glob

from random import randint
from random import uniform

import smach
import smach_ros

import tf
from tf import transformations as tra
import numpy as np

from geometry_msgs.msg import PoseStamped
from subprocess import call
from hybrid_automaton_msgs import srv
from hybrid_automaton_msgs.msg import HAMState

from pregrasp_msgs.msg import GraspStrategyArray
from pregrasp_msgs.msg import GraspStrategy

import hatools.components as ha
import hatools.cookbook as cookbook

def noisy_evaluation(theta):
    # call the other python script
    print(theta)
    
    # write down the reward
    rew = raw_input("What's the reward? ")
    return float(rew)

if __name__ == '__main__':
    '''
    The noisy cross-entropy method consists of three steps:
    (1) Sample from your parameter distribution N(\mu_theta, \sigma_theta)
    (2) Evaluate each parameter sample, collect returns
    (3) Keep n-best samples and infer distribution params
    '''
    parser = argparse.ArgumentParser(description='Cross-Entropy Method')
    
    parser.add_argument('--n_iter', type=int, default = 10,
                        help='Number of iterations of CEM.')
    parser.add_argument('--batch_size', type=int, default = 5,
                        help='Number of samples per batch.')
    parser.add_argument('--elite_frac', type=float, default = 0.2,
                        help='Fraction of samples used as elite set.')
    parser.add_argument('--const_noise', type=float, default = 0.0,
                        help='Additive constant noise.')
    parser.add_argument('--noise_file', type=str, default = 0.0,
                        help='Load this yaml and take the values of key action_std as noise times the const_noise.')
    
    actions = ['sample', 'fit']
    parser.add_argument('action', choices=actions,
                        help='Sample: Generate parameters given a distribution. Fit: Generate normal distribution parameters given the last round samples.')
    parser.add_argument('input_file_or_dir', type=str, 
                        help='File name of the input (directory or file)')
    parser.add_argument('--input_params', type=str, 
                        help='File name of the input (directory or file)')
    parser.add_argument('output_file', type=str, 
                        help='File name of the output (file with distribution params or commands)')
    args = parser.parse_args()
    
    if args.action == 'sample':
        # read in distribution params
        params = yaml.load(open(args.input_file_or_dir, 'r'))
        theta_mean = params['action_mean']
        theta_std = params['action_std']
        
        # Sample parameter vectors
        thetas = np.random.normal(theta_mean, theta_std, (args.batch_size, len(theta_mean)))
        thetas = np.clip(thetas, params['action_min'], params['action_max'])
        
        f = open(args.output_file, 'w')
        
        for i, theta in enumerate(thetas):
            msg = "Run {}:".format(i)
            print(msg)
            f.write(msg + '\n python ec_grasps.py --grasp wall_grasp')
            for id, t in zip(params['action_args'], theta):
                msg = " --{} {}".format(id, t)
                print(msg),
                f.write(msg)
            msg = " " + os.path.dirname(args.output_file)
            print(msg + "")
            f.write(msg + '\n')
        
        f.close()
        
    elif args.action == 'fit':
        # read in old runs
        old_params = yaml.load(open(args.input_params, 'r'))
        
        # read in all runs
        all_runs = glob.glob(args.input_file_or_dir + "/*.yml")
        del all_runs[all_runs.index(args.input_params)]
        
        batch_size = len(all_runs)
        rewards = []
        thetas = []
        for run in all_runs:
            result = yaml.load(open(run, 'r'))
            theta = []
            for id in old_params['action_ids']:
                theta.append(result[id])  # TODO: map arg to 
            thetas.append(np.array(theta))
            rewards.append(float(result['label']))
        
        # Get elite parameters
        n_elite = int(batch_size * args.elite_frac)
        elite_inds = np.argsort(rewards)[batch_size - n_elite:batch_size]
        elite_thetas = [thetas[i] for i in elite_inds]
        
        # Update theta_mean, theta_std (noisy version)
        theta_mean = np.mean(elite_thetas, axis = 0)
        noise = np.ones(theta_mean.shape[0]) * args.const_noise
        if (args.noise_file):
            noise_yaml = yaml.load(open(args.noise_file, 'r'))
            noise = noise_yaml['action_std'] * args.const_noise
        theta_std = np.std(elite_thetas, axis = 0) + noise

        tmp = old_params.copy()
        tmp['action_mean'] = theta_mean
        tmp['action_std'] = theta_std
        yaml.dump(tmp, open(args.output_file, 'w'))
        
        print "Elite frac: ", args.elite_frac
        print "Noise: ", noise
        print "Rewards: ", rewards
        print "Mean reward: %8.3g. Max rewards: %8.3g"%(np.mean(rewards), np.max(rewards))
        print "New theta_mean: ", theta_mean
        print "New theta_std: ", theta_std
        print "Actions: ", tmp['action_ids']
