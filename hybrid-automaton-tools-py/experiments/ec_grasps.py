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

def get_numpy_matrix(listener, pose):
    return listener.fromTranslationRotation((pose.position.x, pose.position.y, pose.position.z),
                                            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))

class TouchFile(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['label_in', 'counter_in'])
    
    def execute(self, userdata):
        user_input = raw_input('Was the last trial a [s]uccess or [f]ailure? ')
        label = 'FAILURE'
        
        if user_input == 's':
            label = 'SUCCESS'
        
        call(['touch', userdata.label_in + '_' + str(userdata.counter_in) + '.' + label + '.label'])
        
        return 'done'
    
class LabelTrial(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['experimental_params_in'],
                             output_keys=['experimental_params_out'])

    def execute(self, userdata):
        label = raw_input('Was this successful? (0-1): ')
        tmp = userdata.experimental_params_in
        tmp['label'] = label
        userdata.experimental_params_out = tmp
        
        return 'done'

class StartBagRecording(smach.State):
    def __init__(self, topics, prefix = '', suffix = ''):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['label_in', 'counter_in', 'pid_in'],
                             output_keys=['pid_out'])
        self.topics = ' '.join(topics)
        self.prefix = prefix
        self.suffix = suffix

    def execute(self, userdata):
        #bag_filename = self.prefix + userdata.label_in + "_" + str(userdata.counter_in) + ".bag"
        bag_filename = self.prefix + userdata.label_in + self.suffix + ".bag"
        command = "rosbag record -O " + bag_filename + " " + self.topics
        p = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        userdata.pid_out = userdata.pid_in + [p.pid]
        return 'done'

class GetGrasp(smach.State):
    def __init__(self, grasp_type, robot_base_frame):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['experimental_params_in'],
                             output_keys=['experimental_params_out'])
        
        self.grasp_type = grasp_type
        self.robot_base_frame = robot_base_frame
        self.grasp_type_map = {
            "wall_grasp": (GraspStrategy.PREGRASP_HOOK, GraspStrategy.STRATEGY_APPROACH_THEN_SQUEEZE),
            "edge_grasp": (GraspStrategy.PREGRASP_HOOK, GraspStrategy.STRATEGY_PUSH),
            "surface_grasp": (GraspStrategy.PREGRASP_SPHERE, GraspStrategy.STRATEGY_SQUEEZE),
        }
        
        assert(grasp_type in self.grasp_type_map.keys() or grasp_type == 'all')
        
        self.br = tf.TransformBroadcaster()
        self.li = tf.TransformListener()
    
    def execute(self, userdata):
        grasp = None
        while grasp is None:
            # get grasp frame and pre-grasp frame
            grasps = rospy.wait_for_message('all_grasps', GraspStrategyArray)
            
            print "No of grasps: ", len(grasps.strategies)
            for g in grasps.strategies:
                if self.grasp_type == 'all' or (g.pregrasp_configuration, g.strategy) == self.grasp_type_map[self.grasp_type]:
                    grasp = g
                    break
            
            rospy.sleep(0.3)
        
        frame_data = []
        frame_data_camera = [] # the same as above but in the local camera frame -- not in the robot base frame
        experimental_params = userdata.experimental_params_in
        
        if self.li.frameExists(self.robot_base_frame):
            if (grasp.pregrasp_configuration, grasp.strategy) == self.grasp_type_map["wall_grasp"]:
                #if len(grasps.strategies) > 0:
                #    grasp = grasps.strategies[randint(0, len(grasps.strategies) - 1)]
                
                experimental_params["type"] = "wall_grasp"
                
                p = grasp.pregrasp_pose.pose
                t = get_numpy_matrix(self.li, p.pose)
                
                # express the target in the robot's base frame
                self.li.waitForTransform(self.robot_base_frame, p.header.frame_id, rospy.Time(), rospy.Duration(10.0))
                p.header.stamp = rospy.Time()
                p_robotbaseframe = self.li.transformPose(self.robot_base_frame, p)
                t_robotbaseframe = get_numpy_matrix(self.li, p_robotbaseframe.pose)
                
                offset_above_table = tra.translation_matrix([experimental_params['position_offset_x'], 0.15, 0.15])
                offset_inside_table = tra.translation_matrix([experimental_params['position_offset_x'], 0.15, -0.2])
                
                frame_data_camera = [
                    t,
                    np.dot(t, offset_above_table),
                    np.dot(t, offset_inside_table)
                ]
                
                frame_data = [
                    np.dot(t_robotbaseframe, offset_above_table),
                    np.dot(t_robotbaseframe, offset_inside_table)
                ]
                
            elif (grasp.pregrasp_configuration, grasp.strategy) == self.grasp_type_map["edge_grasp"]:
                experimental_params["type"] = "edge_grasp"
                
                p0 = grasp.pregrasp_pose.pose
                p1 = grasp.object.pose
                t0 = get_numpy_matrix(self.li, p0.pose)
                t1 = get_numpy_matrix(self.li, p0.pose)
                
                self.li.waitForTransform(self.robot_base_frame, p0.header.frame_id, rospy.Time(), rospy.Duration(10.0))
                p0.header.stamp = rospy.Time()
                p0_robotbaseframe = self.li.transformPose(self.robot_base_frame, p0)
                p1.header.stamp = rospy.Time()
                p1_robotbaseframe = self.li.transformPose(self.robot_base_frame, p1)
                t0_robotbaseframe = get_numpy_matrix(self.li, p0_robotbaseframe.pose)
                t1_robotbaseframe = get_numpy_matrix(self.li, p1_robotbaseframe.pose)
                
                offset_above_table = tra.translation_matrix([0, 0.0, -0.12])
                
                # distance from point t0_mat[:3,3] to the plane spanned by t1_mat[:3,0] and t1_mat[:3,2]
                t0_in_1 = np.dot(tra.inverse_matrix(t0_robotbaseframe), t1_robotbaseframe)
                distance = np.linalg.norm(t0_in_1[:3,3])
                print distance
                
                frame_data = [
                    np.dot(t0_robotbaseframe, offset_above_table),
                    t1_robotbaseframe,
                    distance
                ]
                
                frame_data_camera = [
                    t0,
                    np.dot(t0, offset_above_table),
                    t1,
                    distance
                ]
                
            elif (grasp.pregrasp_configuration, grasp.strategy) == self.grasp_type_map["surface_grasp"]:
                experimental_params["type"] = "surface_grasp"
                p = grasp.pregrasp_pose.pose
                t = get_numpy_matrix(self.li, p.pose)
                
                self.li.waitForTransform(self.robot_base_frame, p.header.frame_id, rospy.Time(), rospy.Duration(10.0))
                p.header.stamp = rospy.Time()
                p_robotbaseframe = self.li.transformPose(self.robot_base_frame, p)
                t_robotbaseframe = get_numpy_matrix(self.li, p_robotbaseframe.pose)
                
                #offset_1 = tra.translation_matrix([0.03, 0.05, -0.17])
                offset_1 = tra.translation_matrix([0.0, 0.0, -0.14])
                offset_2 = tra.translation_matrix([0, 0, +0.1])
                                
                frame_data = [
                    np.dot(t_robotbaseframe, offset_1),
                    np.dot(t_robotbaseframe, offset_2)
                ]
                
                frame_data_camera = [
                    t,
                    np.dot(t, offset_1),
                    np.dot(t, offset_2)
                ]
        
        # publish in ros as tf -- just for checking in rviz
        for i, t in enumerate(frame_data):
            if t.shape == (4, 4):
                self.br.sendTransform( tra.translation_from_matrix(t), tra.quaternion_from_matrix(t), rospy.Time.now(), "/target_dbg_%i" % (i), self.robot_base_frame)

        experimental_params['frames'] = frame_data
        experimental_params['frames_camera'] = frame_data_camera
        userdata.experimental_params_out = experimental_params
        
        return 'done'

class ExecuteMotion(smach.State):
    def __init__(self, experimental_params):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['label_in', 'counter_in', 'motion_params_in'])
        self.call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        self.experimental_params = experimental_params
    
    def execute(self, userdata):
        if userdata.motion_params_in['type'] == 'wall_grasp':
            # Good values
            # python ec_grasps.py --angle 65.0 --inflation 0.07 --speed 0.04 --force 3.5 --wallforce -12.0 --grasp wall_grasp test_folder
            # python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp wall_grasp wall_chewinggum
            print 'Executing WALL GRASP'
            curree = userdata.motion_params_in['frames'][0]
            #curree = np.dot(curree, tra.rotation_matrix(self.experimental_params[userdata.counter_in], [1, 0, 0]))
            curree = np.dot(curree, tra.rotation_matrix(self.experimental_params['angle_of_attack'], [1, 0, 0]))
            reposition_above_table = ha.HTransformControlMode(curree, name = 'reposition', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            goal = np.dot(tra.translation_matrix([0, 0, -0.3]), curree)
            go_down = ha.HTransformControlMode(goal, name = 'go_down', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            slide = ha.ControlMode(name = 'slide').set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.25, desired_displacement=tra.translation_matrix([self.experimental_params['sliding_speed'], 0, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0]))))
            close_hand = ha.GravityCompensationMode(name = 'close_hand')
            hand_closing_time = 2.5
            #close_hand.controlset.add(ha.RBOHandController(goal = np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), valve_times = np.array([[0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time]]), goal_is_relative = '1'))
            # including thumb
            close_hand.controlset.add(ha.RBOHandController(goal = np.array([[1,0]]*6), valve_times = np.array([[0,hand_closing_time]]*6), goal_is_relative = '1'))
            retract = ha.HTransformControlMode(curree, name = 'retract', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            
            ee_switch = ha.FramePoseSwitch('reposition', 'go_down', controller = 'GoToCartesianConfig', epsilon = '0.02')
            ft_switch = ha.ForceTorqueSwitch('go_down', 'slide', goal = np.array([0, 0, self.experimental_params['downward_force'], 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1')
            #distance_switch = ha.FrameDisplacementSwitch('slide', 'finished', epsilon = '0.15', negate = '1', goal = np.array([0, 0, 0]), goal_is_relative = '1', jump_criterion = "NORM_L2", frame_id = 'EE')
            #wall_ft_switch.add(distance_switch.conditions[0])
            wall_ft_switch = ha.ForceTorqueSwitch('slide', 'close_hand', goal = np.array([self.experimental_params['wall_force'], 0, 0, 0, 0, 0]), norm_weights = np.array([1, 0, 0, 0, 0, 0]), jump_criterion = "THRESH_LOWER_BOUND", frame_id = 'odom', goal_is_relative = '1')
            hand_closing_switch = ha.TimeSwitch('close_hand', 'retract', duration = 1.0 + hand_closing_time)
            retract_switch = ha.FramePoseSwitch('retract', 'finished', controller = 'GoToCartesianConfig', epsilon = '0.01')
            
            finished = ha.GravityCompensationMode(name = 'finished')
            
            myha = ha.HybridAutomaton(current_control_mode='reposition').add([reposition_above_table, ee_switch, go_down, ft_switch, slide, wall_ft_switch, close_hand, hand_closing_switch, retract, retract_switch, finished])
        elif userdata.motion_params_in['type'] == 'edge_grasp':
            # Good values
            # python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp edge_grasp --edgedistance -0.013 test_folder
            # python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp edge_grasp --edgedistance -0.007 edge_chewinggum/
            print 'Executing EDGE GRASP'
            initial_cspace_goal = np.array([0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])
            go_above_table = ha.JointControlMode(initial_cspace_goal, name = 'go_above_table', controller_name = 'GoToJointConfig')

            curree = userdata.motion_params_in['frames'][0]
            curree = np.dot(curree, tra.rotation_matrix(self.experimental_params['angle_of_sliding'], [1, 0, 0]))
            reposition_above_table = ha.HTransformControlMode(curree, name = 'reposition', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            goal = np.dot(tra.translation_matrix([0, 0, -0.3]), curree)
            go_down = ha.HTransformControlMode(goal, name = 'go_down', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            slide = ha.ControlMode(name = 'slide').set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.25, desired_displacement=tra.translation_matrix([0, -self.experimental_params['sliding_speed'], 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0]))))
            close_hand = ha.GravityCompensationMode(name = 'close_hand')
            hand_closing_time = 3.0
            close_hand.controlset.add(ha.RBOHandController(goal = np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), valve_times = np.array([[0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time]]), goal_is_relative = '1'))
            #retract = ha.HTransformControlMode(curree, name = 'retract', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            retract = ha.HTransformControlMode(tra.translation_matrix([0, 0, -0.1]), name = 'retract', controller_name = 'GoToCartesianConfig', goal_is_relative='1')
            
            joint_switch = ha.JointConfigurationSwitch('go_above_table', 'reposition', controller = 'GoToJointConfig', epsilon = str(math.radians(7.0)))
            ee_switch = ha.FramePoseSwitch('reposition', 'go_down', controller = 'GoToCartesianConfig', epsilon = '0.001')
            ft_switch = ha.ForceTorqueSwitch('go_down', 'slide', goal = np.array([0, 0, self.experimental_params['downward_force'], 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1')
            distance_switch = ha.FrameDisplacementSwitch('slide', 'close_hand', epsilon = str(self.experimental_params['edge_distance_factor'] + userdata.motion_params_in['frames'][-1]), negate = '1', goal = np.array([0, 0, 0]), goal_is_relative = '1', jump_criterion = "NORM_L2", frame_id = 'EE')
            #wall_ft_switch = ha.ForceTorqueSwitch('slide', 'close_hand', goal = np.array([self.experimental_params['wall_force'], 0, 0, 0, 0, 0]), norm_weights = np.array([1, 0, 0, 0, 0, 0]), jump_criterion = "THRESH_LOWER_BOUND", frame_id = 'odom', goal_is_relative = '1')
            #wall_ft_switch.add(distance_switch.conditions[0])
            #time_switch = ha.TimeSwitch('slide', 'finished', duration = 1.0)
            hand_closing_switch = ha.TimeSwitch('close_hand', 'retract', duration = 1.0 + hand_closing_time)
            #retract_switch = ha.FramePoseSwitch('retract', 'finished', controller = 'GoToCartesianConfig', epsilon = '0.001')
            retract_switch = ha.FramePoseSwitch('retract', 'finished', controller = 'GoToCartesianConfig', epsilon = '0.001', goal_is_relative = '1')
            
            finished = ha.GravityCompensationMode(name = 'finished')
            
            #wall_ft_switch, close_hand, hand_closing_switch, retract, retract_switch, 
            myha = ha.HybridAutomaton(current_control_mode='go_above_table').add([go_above_table, joint_switch, reposition_above_table, ee_switch, go_down, ft_switch, slide, distance_switch, close_hand, hand_closing_switch, retract, retract_switch, finished])
        elif userdata.motion_params_in['type'] == 'surface_grasp':
            # Good values
            # python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp surface_grasp test_folder
            print 'EXECUTE SURFACE GRASP'
            goals = userdata.motion_params_in['frames']
            goals = [np.dot(g, tra.translation_matrix([0, 0.02, 0])) for g in goals]
            goals = [np.dot(g, tra.rotation_matrix(self.experimental_params['angle_of_sliding'], [1, 0, 0])) for g in goals]
            
            initial_cspace_goal = np.array([0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])
            go_above_table = ha.JointControlMode(initial_cspace_goal, name = 'go_above_table', controller_name = 'GoToJointConfig')
            move_into_table = ha.HTransformControlMode(np.hstack(goals), name = 'move_into_table', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            #close_hand = ha.GravityCompensationMode(name = 'close_hand')  # is not really smooth (and force is lost..)
            #close_hand = ha.JointControlMode(np.zeros(7), name = 'close_hand', goal_is_relative='1') # is not really smooth
            close_hand = ha.ControlMode(name = 'close_hand').set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.0, desired_displacement=tra.translation_matrix([0, 0, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0]))))
            # hand_closing_time = 2.5   # normal RBO hand 2
            valve_times_times = np.array([[ 0. ,  4.1],
                                            [ 0. ,  0.1],
                                            [ 0. ,  5. ],
                                            [ 0. ,  5.],
                                            [ 0. ,  2.],
                                            [ 0. ,  3.5]])
            hand_closing_time = np.max(valve_times_times)#3.0 # special palm prototype
            #valve_times_times = np.vstack([[0,hand_closing_time]]*6)
            close_hand.controlset.add(ha.RBOHandController(goal = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]), valve_times = valve_times_times, goal_is_relative = '1'))
            retract = ha.HTransformControlMode(goals[0], name = 'retract', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
            finished = ha.GravityCompensationMode(name = 'finished')
            
            joint_switch = ha.JointConfigurationSwitch('go_above_table', 'move_into_table', controller = 'GoToJointConfig', epsilon = str(math.radians(7.)))
            ft_switch = ha.ForceTorqueSwitch('move_into_table', 'close_hand', goal = np.array([0, 0, self.experimental_params['downward_force'], 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1')
            hand_closing_switch = ha.TimeSwitch('close_hand', 'retract', duration = 1.0 + hand_closing_time)
            retract_switch = ha.FramePoseSwitch('retract', 'finished', controller = 'GoToCartesianConfig', epsilon = '0.001')
            
            myha = ha.HybridAutomaton(current_control_mode='go_above_table').add([go_above_table, joint_switch, move_into_table, ft_switch, close_hand, hand_closing_switch, retract, retract_switch, finished])
        
        #print myha.xml()
        self.call_ha(myha.xml())
        
        return 'done'

class KillPG(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['pid_in'],
                             output_keys=['pid_out'])

    def execute(self, userdata):
        if len(userdata.pid_in) > 0:
            print "killing! "
            for pid in userdata.pid_in:
                print pid
                os.killpg(pid, signal.SIGINT)
            userdata.pid_out = []
        else:
            print "Nothing to kill!"
        return 'done'

class ReturnToStart(smach.State):
    def __init__(self, motion_params):
        smach.State.__init__(self, outcomes=['done'])
        self.call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        
        jgoal = np.array([0.236849, -0.974282, -2.11068, 2.6053, -0.386705, 0.859499, 0.742537])
        go_above_table = ha.JointControlMode(jgoal, name = 'go_above_table', controller_name = 'GoToJointConfig')
        # that's important to avoid jerky behavior on the spot! --> it's actually the minimum_completion_time
        go_above_table.controlset.controllers[0].properties['completion_times'] = '[1,1]1.0'
        if motion_params['finger_inflation'] > 0:
            go_above_table.controlset.add(ha.RBOHandController(goal = np.array([[-1,0,0],[-1,0,0],[-1,1,0],[-1,1,0],[-1,1,0],[-1,1,0]]), valve_times = np.array([[0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']]]), goal_is_relative = '1'))
        joint_switch = ha.JointConfigurationSwitch('go_above_table', 'waiting', controller = 'GoToJointConfig', epsilon = str(math.radians(10.0)))
        time_switch = ha.TimeSwitch('go_above_table', 'waiting', duration = 20.0)
        joint_switch.add(ha.JumpCondition('ClockSensor', goal = np.array([[5.0 + 2.0 * motion_params['finger_inflation']]]), jump_criterion = 'THRESH_UPPER_BOUND', goal_is_relative = '1', epsilon = '0'))
        waiting = ha.GravityCompensationMode(name = 'waiting')
                
        self.return_ha = ha.HybridAutomaton(current_control_mode='go_above_table').add([go_above_table, joint_switch, time_switch, waiting])
        

    def execute(self, userdata):
        #print self.return_ha.xml()
        self.call_ha(self.return_ha.xml())
        return 'done'

class WaitForHAMState(smach.State):
    def __init__(self, desired_ham_states):
        smach.State.__init__(self, outcomes=desired_ham_states)
        
        self.current_ham_state = "Undefined"
        self.desired_ham_states = desired_ham_states
        rospy.Subscriber("/ham_state", HAMState, self.ham_callback)
    
    def ham_callback(self, msg):
        self.current_ham_state = msg.executing_control_mode_name
    
    def execute(self, userdata):
        self.current_ham_state = "Undefined"
        
        while self.current_ham_state not in self.desired_ham_states:
            rospy.sleep(0.3)
        
        return self.current_ham_state

class Wait(smach.State):
    def __init__(self, duration):
        smach.State.__init__(self, outcomes=['done'])
        self.duration = duration
    
    def execute(self, userdata):
        rospy.sleep(self.duration)
        return 'done'

class GatherTFData(smach.State):
    def __init__(self, tfs):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['experimental_params_in'],
                             output_keys=['experimental_params_out'])
        self.tfs = tfs
        self.listener = tf.TransformListener()
    
    def execute(self, userdata):
        frames = []
        for frame_id, parent in self.tfs:
            if not self.listener.frameExists(frame_id):
                frames.append((frame_id, parent, "TF does not exist!"))
                continue
            if not self.listener.frameExists(parent):
                frames.append((frame_id, parent, "Parent does not exist!"))
                continue
            now = rospy.Time.now()
            self.listener.waitForTransform(parent, frame_id, now, rospy.Duration(10.0))
            (trans,rot) = self.listener.lookupTransform(parent, frame_id, now)
            T = self.listener.fromTranslationRotation(trans, rot)
            frames.append((frame_id, parent, T))
        
        old_params = userdata.experimental_params_in
        old_params['transformations'] = frames
        userdata.experimental_params_out = old_params
        
        return 'done'

class RecordPCD(smach.State):
    def __init__(self, record_pcd = True, record_png = True, prefix = '', suffix = '', pcd_topic = None, rgb_topic = None):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['label_in', 'counter_in','experimental_params_in'],
                             output_keys=['experimental_params_out'])
        self.prefix = prefix
        self.suffix = suffix
        self.record_pcd = record_pcd
        self.record_png = record_png
        self.pcd_topic = pcd_topic
        self.rgb_topic = rgb_topic

    def execute(self, userdata):
        rgbd_filename = userdata.label_in + self.suffix# + str(userdata.counter_in)
        cmd = ['rosrun', 'rbo_utils', 'single_pointcloud_to_pcd']
        
        tmp = userdata.experimental_params_in
        
        if self.record_pcd:
            tmp['pcd_file' + self.suffix] = rgbd_filename + '.pcd'
            cmd += ['_pcd_filename:=' + self.prefix + rgbd_filename + '.pcd']
            if self.pcd_topic is not None:
                cmd += ['_camera_pointcloud_topic:=' + self.pcd_topic]
        if self.record_png:
            tmp['png_file' + self.suffix] = rgbd_filename + '.png'
            cmd += ['_rgb_png_filename:=' + self.prefix + rgbd_filename + '.png']
            if self.rgb_topic is not None:
                cmd += ['_camera_rgb_topic:=' + self.rgb_topic]
        
        call(cmd)
        userdata.experimental_params_out = tmp
        
        return 'done'


def main(**args):
    if not os.path.isdir(args['directory']):
        print "Directory '%s' does not exist!" % (args['directory'])
        sys.exit()
    
    experiment_label = datetime.datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    rospy.init_node('top_down_grasps_with_different_shapes')

    experimental_parameters = {}
    experimental_parameters['grasp_type'] = args['grasp']
    experimental_parameters['angle_of_attack'] = math.radians(args['angle']) if args['angle'] > 2.0 else args['angle']
    experimental_parameters['position_offset_x'] = args['positionx']
    experimental_parameters['sliding_speed'] = args['speed']
    experimental_parameters['finger_inflation'] = args['inflation']
    experimental_parameters['downward_force'] = args['force']
    experimental_parameters['wall_force'] = args['wallforce']
    experimental_parameters['angle_of_sliding'] = math.radians(args['anglesliding'])
    experimental_parameters['edge_distance_factor'] = args['edgedistance']
    
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['FINISHED'])
    sm.userdata.trial_label = experiment_label #args.label
    sm.userdata.trial_number = 0 #args.num
    sm.userdata.pid = []
    sm.userdata.experimental_parameters = experimental_parameters
    
    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('RETURN_TO_START', ReturnToStart(experimental_parameters), 
                               transitions={'done':'WAIT_FOR_HAM_STATE_INIT'})
        smach.StateMachine.add('WAIT_FOR_HAM_STATE_INIT', WaitForHAMState(['waiting', 'GravityCompensation']),
                               transitions={'waiting':'WAIT_A_LITTLE_WHILE',
                                            'GravityCompensation': 'FINISHED'})
        smach.StateMachine.add('WAIT_A_LITTLE_WHILE', Wait(3.0),
                               transitions={'done':'GATHER_TF_DATA'})
                               
        smach.StateMachine.add('GATHER_TF_DATA', GatherTFData([('table', 'camera_rgb_optical_frame'), #('wall', 'camera_rgb_optical_frame'),
                                                               ('camera_rgb_optical_frame', args['baseframe'])]),
                               remapping={'experimental_params_in': 'experimental_parameters',
                                          'experimental_params_out': 'experimental_parameters'},
                               transitions={'done':'RECORD_PCD'})
        
        smach.StateMachine.add('RECORD_PCD', RecordPCD(prefix = args['directory'] + '/', suffix = "_before"),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number',
                                          'experimental_params_in': 'experimental_parameters',
                                          'experimental_params_out': 'experimental_parameters'},
                               transitions={'done':'GET_GRASP'})
        
        smach.StateMachine.add('GET_GRASP', GetGrasp(args['grasp'], args['baseframe']),
                               remapping={'experimental_params_in': 'experimental_parameters',
                                          'experimental_params_out': 'experimental_parameters'},
                               transitions={'done':'START_BAG_RECORDING'})
        
        smach.StateMachine.add('START_BAG_RECORDING', StartBagRecording(["/ee_velocity", "/joint_states", "/pressures", "/tf", "/ft_sensor", "/ft_world", "/ham_state"], prefix = args['directory'] + '/'),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number',
                                          'pid_in':'pid',
                                          'pid_out':'pid'},
                               transitions={'done':'EXECUTE_MOTION'})
        
        smach.StateMachine.add('START_VIDEO_RECORDING', StartBagRecording(["/camera1/image_color", "/camera2/image_color"], prefix = args['directory'] + "/", suffix = "_video"),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number',
                                          'pid_in':'pid',
                                          'pid_out':'pid'},
                               transitions={'done':'EXECUTE_MOTION'})
        
        smach.StateMachine.add('EXECUTE_MOTION', ExecuteMotion(experimental_parameters),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number',
                                          'motion_params_in':'experimental_parameters'},
                               transitions={'done':'WAIT_FOR_HAM_STATE'})
        
        smach.StateMachine.add('WAIT_FOR_HAM_STATE', WaitForHAMState(['finished', 'GravityCompensation']),
                               transitions={'finished': 'STOP_BAG_RECORDINGS',
                                            'GravityCompensation': 'STOP_BAG_RECORDINGS'})
        smach.StateMachine.add('STOP_BAG_RECORDINGS', KillPG(),
                               remapping={'pid_in':'pid',
                                          'pid_out':'pid'},
                               transitions={'done':'FINISHED'})
        
        smach.StateMachine.add('LABEL_TRIAL', LabelTrial(),
                               remapping={'experimental_params_in': 'experimental_parameters',
                                          'experimental_params_out': 'experimental_parameters'},
                               transitions={'done':'RECORD_PNG'})
        
        smach.StateMachine.add('RECORD_PNG', RecordPCD(prefix = args['directory'] + '/', suffix = "_after", record_pcd = False, rgb_topic='/camera1/image_color'),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number',
                                          'experimental_params_in': 'experimental_parameters',
                                          'experimental_params_out': 'experimental_parameters'},
                               transitions={'done':'FINISHED'})
    
    sm.set_initial_state(['RETURN_TO_START'])
    
    # Execute SMACH plan
    outcome = sm.execute()
    
    if outcome is "FINISHED":    
        with open(args['directory'] + '/' + experiment_label + '.yml', 'w') as outfile:
            outfile.write( yaml.dump(experimental_parameters, default_flow_style=False) )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record sliding wall grasps.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('directory', type=str, default='foo',
                        help='the directory to write files to')
    parser.add_argument('--baseframe', type=str, default='odom',
                        help='usually either odom or base_link, or camera_link if no robot is active')
    parser.add_argument('--angle', type=float, default = 10.,
                        help='angle of attack during sliding (in deg)')
    parser.add_argument('--speed', type=float, default = 0.01,
                        help='speed during sliding')
    parser.add_argument('--inflation', type=float, default = 0.05,
                        help='inflation of fingers during sliding')
    parser.add_argument('--force', type=float, default = 4.,
                        help='downward force during sliding')
    parser.add_argument('--wallforce', type=float, default = -6.,
                        help='wall force to finish sliding')
    parser.add_argument('--anglesliding', type=float, default = -10.,
                        help='angle during sliding to edge (in deg)')
    parser.add_argument('--edgedistance', type=float, default = 1.,
                        help='multiplicative factor for the jump condition for the distance to the edge')
    parser.add_argument('--positionx', type=float, default = 0.,
                        help='positional offset parallel to the constraint (e.g. wall)')
    grasps = ["all", "edge_grasp", "wall_grasp", "surface_grasp"]
    parser.add_argument('--grasp', choices=grasps, default=grasps[0],
                        help='which grasp type to use, default: all')
                        
    #parser.add_argument('--label', type=str, default="foo",
    #                    help='the object label to start with')
    #parser.add_argument('--num', type=int, default=-1,
    #                    help='the trial number to start with')
    args = parser.parse_args()
    
    main(**vars(args))
