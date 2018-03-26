#!/usr/bin/python

# Simple script that creates a hybrid automaton to move the 10 dof robot, pauses it and moves it again
# We used this script to find and debug the interpolation bug (using always initial velocity = 0 
# instead of the real initial velocity)

import rospy
from hybrid_automaton_msgs import srv
call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)

import sys

#Change the following path to point to the location of your hybrid automaton tools
path_to_hat = '/home/roberto/Libraries/hybrid-automaton-tools-py'
sys.path.append(path_to_hat)

from hatools import components, cookbook, rosbags, utils
import hatools.components as ha

import numpy as np

# Lists of control modes and switches
control_modes = []
control_switches = []

# Initial control mode
initial = ha.GravityCompensationMode(name='initial_cm')
control_modes.extend([initial])


# Final control mode
finished = ha.GravityCompensationMode(name='finished')
control_modes.extend([finished])

# Failure control mode
failure = ha.GravityCompensationMode(name='failure')
control_modes.extend([failure])

# EE Goal: only translation or translation+rotation
homog_transf = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-0.1],[0,0,0,1]])
#homog_transf = np.array([[1,0,0,0],[0,0.8660254, 0.5,0],[0,-0.5,0.8660254,0],[0,0,0,1]])

# Trajectory control mode
traj_cm = ha.HTransformControlMode(homog_transf, name='traj_cm', controller_name = 'traj_ctrl', goal_is_relative='1',
                                           completion_times = np.array([5]), joint_weights = np.array([1,1,1,1,1,1,1,1,1,1]))
traj_cm.controlset.properties['js_kp'] = np.array([30,20,15,20,10,10,10,0,0,0])
traj_cm.controlset.properties['js_kd'] = np.array([1,2,1,0.4,0.1,0.1,0.01,6,6,6])
#traj_cm.controlset.controllers[0].properties['type'] = "HTransformController"
control_modes.extend([traj_cm])

# Control switch INTO the trajectory control mode (time, 0.1 secs)
switch = ha.TimeSwitch('initial_cm', 'traj_cm', name='start_motion_cs', duration=0.1)
control_switches.extend([switch])
                                           
# Gravity compensation pause after the first part of the trajectory
pause = ha.GravityCompensationMode(name='pause')
control_modes.extend([pause])

# Control switch INTO the gravity compensation cm
switch_pause = ha.TimeSwitch('traj_cm', 'pause', name='pause_cs', duration=5.)
#switch_pause = ha.FramePoseSwitch('traj_cm', 'pause',name='pause_cs', controller='traj_ctrl', epsilon='0.1')
control_switches.extend([switch_pause])

# Second part of the trejectory
traj2_cm = ha.HTransformControlMode(homog_transf, name='traj2_cm', controller_name = 'traj2_ctrl', goal_is_relative='1',
                                           completion_times = np.array([5]), joint_weights = np.array([1,1,1,1,1,1,1,1,1,1]))
traj2_cm.controlset.properties['js_kp'] = np.array([30,20,15,20,10,10,10,0,0,0])
traj2_cm.controlset.properties['js_kd'] = np.array([1,2,1,0.4,0.1,0.1,0.01,6,6,6])
#traj2_cm.controlset.controllers[0].properties['type'] = "HTransformController"
control_modes.extend([traj2_cm])

# Control switch INTO the second part of the trajectory
time_s = ha.TimeSwitch('pause', 'traj2_cm', name='motion_ended', duration=0.1)
control_switches.extend([time_s])

# Create hybrid automaton and send it
myha = ha.HybridAutomaton(current_control_mode='initial_cm').add(control_modes + control_switches)
print myha.xml()
call_ha(myha.xml())

#switch_new = ha.FramePoseSwitch('traj_cm', 'finished',name='finishing_cs', controller='traj_ctrl', epsilon='0.05')
#control_switches.extend([switch_new])

#safety_jv = 0.4
#jv_switch = ha.JointVelocitySwitch('traj_cm', 'failure',name='exceeded_jv_cs',
                                           #norm_weights=np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
                                           #goal=np.array([0, 0, 0,0, 0, 0, 0, 0, 0, 0]),
                                           #epsilon=str(safety_jv),
                                           #jump_criterion="NORM_L_INF",
                                           #negate=str(1))
                                           
#safety_jv_b = 0.2
#jv_switch_b = ha.JointVelocitySwitch('traj_cm', 'failure',name='exceeded_jv_cs_base',
                                           #norm_weights=np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
                                           #goal=np.array([0, 0, 0,0, 0, 0, 0, 0, 0, 0]),
                                           #epsilon=str(safety_jv_b),
                                           #jump_criterion="NORM_L_INF",
                                           #negate=str(1))
