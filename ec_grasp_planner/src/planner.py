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

from geometry_graph_msgs.msg import Graph

import pyddl

import hatools.components as ha
import hatools.cookbook as cookbook

import hand_parameters

def get_numpy_matrix(listener, pose):
    return listener.fromTranslationRotation((pose.position.x, pose.position.y, pose.position.z),
                                            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))

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
            #close_hand.controlset.add(ha.RBOHandController(goal = np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), valve_opening = np.array([[0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time]]), goal_is_relative = '1'))
            # including thumb
            close_hand.controlset.add(ha.RBOHandController(goal = np.array([[1,0]]*6), valve_opening = np.array([[0,hand_closing_time]]*6), goal_is_relative = '1'))
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
            close_hand.controlset.add(ha.RBOHandController(goal = np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), valve_opening = np.array([[0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time]]), goal_is_relative = '1'))
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
            valve_opening_times = np.array([[ 0. ,  4.1],
                                            [ 0. ,  0.1],
                                            [ 0. ,  5. ],
                                            [ 0. ,  5.],
                                            [ 0. ,  2.],
                                            [ 0. ,  3.5]])
            hand_closing_time = np.max(valve_opening_times)#3.0 # special palm prototype
            #valve_opening_times = np.vstack([[0,hand_closing_time]]*6)
            close_hand.controlset.add(ha.RBOHandController(goal = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]), valve_opening = valve_opening_times, goal_is_relative = '1'))
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

class ReturnToStart(smach.State):
    def __init__(self, motion_params):
        smach.State.__init__(self, outcomes=['done'])
        self.call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        
        jgoal = np.array([0.236849, -0.974282, -2.11068, 2.6053, -0.386705, 0.859499, 0.742537])
        go_above_table = ha.JointControlMode(jgoal, name = 'go_above_table', controller_name = 'GoToJointConfig')
        # that's important to avoid jerky behavior on the spot! --> it's actually the minimum_completion_time
        go_above_table.controlset.controllers[0].properties['completion_times'] = '[1,1]1.0'
        if motion_params['finger_inflation'] > 0:
            go_above_table.controlset.add(ha.RBOHandController(goal = np.array([[-1,0,0],[-1,0,0],[-1,1,0],[-1,1,0],[-1,1,0],[-1,1,0]]), valve_opening = np.array([[0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']]]), goal_is_relative = '1'))
        joint_switch = ha.JointConfigurationSwitch('go_above_table', 'waiting', controller = 'GoToJointConfig', epsilon = str(math.radians(10.0)))
        time_switch = ha.TimeSwitch('go_above_table', 'waiting', duration = 20.0)
        joint_switch.add(ha.JumpCondition('ClockSensor', goal = np.array([[5.0 + 2.0 * motion_params['finger_inflation']]]), jump_criterion = 'THRESH_UPPER_BOUND', goal_is_relative = '1', epsilon = '0'))
        waiting = ha.GravityCompensationMode(name = 'waiting')
                
        self.return_ha = ha.HybridAutomaton(current_control_mode='go_above_table').add([go_above_table, joint_switch, time_switch, waiting])
        

    def execute(self, userdata):
        #print self.return_ha.xml()
        self.call_ha(self.return_ha.xml())
        return 'done'

#def is_goal_node(strategy_type, grasp_type):
#    return (strategy_type == GraspStrategy.STRATEGY_WALL_GRASP or
#            strategy_type == GraspStrategy.STRATEGY_SQUEEZE or
#            strategy_type == GraspStrategy.STRATEGY_APPROACH_THEN_SQUEEZE or
#            strategy_type == GraspStrategy.STRATEGY_SLIDE_TO_EDGE )

def find_all_paths(start_node_id, goal_node_ids, graph):
    solutions = []
    intermediate_solutions = []
    final_solutions = []
    
    p1 = [start_node_id]
    intermediate_solutions.append(p1)
    
    graph_edges = [(e.node_id_start, e.node_id_end) for e in graph.edges]
    
    while True:
        solutions = intermediate_solutions;
        intermediate_solutions = []
        
        for p in solutions:
            node_index = p[-1]
            #if is_goal_node(nodes[node_index].strategy, nodes[node_index].pregrasp_configuration):
            if node_index in goal_node_ids:
                final_solutions.append(p)
            else:
                # go through all neighbors and add for each one a new path
                for j in range(len(graph.nodes)):
                    if j in p: #(std::find(p.begin(), p.end(), j) != p.end())
                        continue
                    
                    if (node_index, j) in graph_edges:
                        new_p = list(p)
                        new_p.append(j)
                        intermediate_solutions.append(new_p)
        
        if len(intermediate_solutions) == 0:
            break
    
    print("No. of solution paths: {}".format(len(final_solutions)));
    
    return final_solutions

def create_wall_grasp(object_frame, support_surface_frame, wall_frame):
    # Good values
    # python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp wall_grasp wall_chewinggum
    
    downward_force = 3.
    sliding_speed = 0.04
    wall_force = -11.0
    angle_of_attack = math.radians(69.0)
    
    curree = object_frame # TODO
    #curree = np.dot(curree, tra.rotation_matrix(self.experimental_params[userdata.counter_in], [1, 0, 0]))
    curree = np.dot(curree, tra.rotation_matrix(angle_of_attack, [1, 0, 0]))
    goal = np.dot(tra.translation_matrix([0, 0, -0.3]), curree)
    
    control_sequence = []
    
    control_sequence.append(ha.HTransformControlMode(curree, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.02'))
    control_sequence.append(ha.HTransformControlMode(goal, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([0, 0, downward_force, 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.ControlMode().set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.25, desired_displacement=tra.translation_matrix([sliding_speed, 0, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0])))))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([wall_force, 0, 0, 0, 0, 0]), norm_weights = np.array([1, 0, 0, 0, 0, 0]), jump_criterion = "THRESH_LOWER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.GravityCompensationMode())
    hand_closing_time = 2.5
    control_sequence[-1].controlset.add(ha.RBOHandController(goal = np.array([[1,0]]*6), valve_opening = np.array([[0,hand_closing_time]]*6), goal_is_relative = '1'))
    control_sequence.append(ha.TimeSwitch('', '', duration = 1.0 + hand_closing_time))
    control_sequence.append(ha.HTransformControlMode(curree, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.01'))
    control_sequence.append(ha.GravityCompensationMode())
    
    return cookbook.sequence_of_modes_and_switches(control_sequence)

def create_edge_grasp(object_frame, support_surface_frame, edge_frame):
    # python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp edge_grasp --edgedistance -0.007 edge_chewinggum/
    edge_distance_factor = -0.007
    distance = 0. # TODO
    downward_force = 4.0
    sliding_speed = 0.04
    angle_of_sliding = math.radians(-10.)

    initial_cspace_goal = np.array([0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])

    curree = userdata.motion_params_in['frames'][0]
    curree = np.dot(curree, tra.rotation_matrix(angle_of_sliding, [1, 0, 0]))
    goal = np.dot(tra.translation_matrix([0, 0, -0.3]), curree)
    hand_closing_time = 3.0
    
    control_sequence = []
    control_sequence.append(ha.HTransformControlMode(curree, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.001'))
    control_sequence.append(ha.HTransformControlMode(goal, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([0, 0, downward_force, 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.ControlMode().set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.25, desired_displacement=tra.translation_matrix([0, -sliding_speed, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0])))))
    control_sequence.append(ha.FrameDisplacementSwitch('', '', epsilon = str(edge_distance_factor + distance), negate = '1', goal = np.array([0, 0, 0]), goal_is_relative = '1', jump_criterion = "NORM_L2", frame_id = 'EE'))
    control_sequence.append(ha.GravityCompensationMode())
    control_sequence[-1].controlset.add(ha.RBOHandController(goal = np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), valve_opening = np.array([[0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time]]), goal_is_relative = '1'))
    control_sequence.append(ha.TimeSwitch('', '', duration = 1.0 + hand_closing_time))
    control_sequence.append(ha.HTransformControlMode(tra.translation_matrix([0, 0, -0.1]), controller_name = 'GoToCartesianConfig', goal_is_relative='1'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.001', goal_is_relative = '1'))
    control_sequence.append(ha.GravityCompensationMode())
    
    return cookbook.sequence_of_modes_and_switches(control_sequence)

def create_surface_grasp(object_frame, support_surface_frame):
    # python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp surface_grasp test_folder
    angle_of_sliding = 0.0
    downward_force = 7.
    
    goals = object_frame # TODO 
    goals = [np.dot(g, tra.translation_matrix([0, 0.02, 0])) for g in goals]
    goals = [np.dot(g, tra.rotation_matrix(angle_of_sliding, [1, 0, 0])) for g in goals]
    
    initial_cspace_goal = np.array([0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])

    valve_opening_times = np.array([[ 0. ,  4.1],
                                    [ 0. ,  0.1],
                                    [ 0. ,  5. ],
                                    [ 0. ,  5.],
                                    [ 0. ,  2.],
                                    [ 0. ,  3.5]])
    hand_closing_time = np.max(valve_opening_times)#3.0 # special palm prototype
    
    control_sequence = []
    control_sequence.append(ha.HTransformControlMode(np.hstack(goals), controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([0, 0, downward_force, 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.ControlMode().set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.0, desired_displacement=tra.translation_matrix([0, 0, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0])))))
    control_sequence[-1].controlset.add(ha.RBOHandController(goal = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]), valve_opening = valve_opening_times, goal_is_relative = '1'))
    control_sequence.append(ha.TimeSwitch('', '', duration = 1.0 + hand_closing_time))
    control_sequence.append(ha.HTransformControlMode(goals[0], controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.001'))
    control_sequence.append(ha.GravityCompensationMode())
            
    return cookbook.sequence_of_modes_and_switches(control_sequence)

def hybrid_automaton_from_motion_sequence(motion_sequence, graph):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))
    
    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label
    grasp_frame = grasp_frames[grasp_type]
    
    # instantiate every motion relative to the object frame
    object_frame = tra.translation_matrix([0, 0, 0])
    
    if grasp_type == 'EdgeGrasp':
        return create_edge_grasp(object_frame, support_surface_frame)
    elif grasp_type == 'WallGrasp':
        return create_wall_grasp(object_frame, support_surface_frame)
    elif grasp_type == 'SurfaceGrasp':
        return create_surface_grasp(object_frame, support_surface_frame)
    else:
        raise "Unknown grasp type: ", grasp_type

def find_a_path(hand_start_node_id, object_start_node_id, graph, goal_node_id = None, verbose = False):
    locations = ['l'+str(i) for i in range(len(graph.nodes))]
    
    connections = [('connected', 'l'+str(e.node_id_start), 'l'+str(e.node_id_end)) for e in graph.edges]
    grasping_locations = [('is_grasping_location', 'l'+str(i)) for i, n in enumerate(graph.nodes) if n.label.endswith('Grasp')]
        
    # define possible actions
    domain = pyddl.Domain((
        pyddl.Action(
            'move_hand',
            parameters=(
                ('location', 'from'),
                ('location', 'to'),
            ),
            preconditions=(
                ('hand_at', 'from'),
                ('connected', 'from', 'to'),
            ),
            effects=(
                pyddl.neg(('hand_at', 'from')),
                ('hand_at', 'to'),
            ),
        ),
        pyddl.Action(
            'move_object',
            parameters=(
                ('location', 'from'),
                ('location', 'to'),
            ),
            preconditions=(
                ('hand_at', 'from'),
                ('object_at', 'from'),
                ('connected', 'from', 'to'),
            ),
            effects=(
                pyddl.neg(('hand_at', 'from')),
                pyddl.neg(('object_at', 'from')),
                ('hand_at', 'to'),
                ('object_at', 'to'),
            ),
        ),
        pyddl.Action(
            'grasp_object',
            parameters=(
                ('location', 'l'),
            ),
            preconditions=(
                ('hand_at', 'l'),
                ('object_at', 'l'),
                ('is_grasping_location', 'l')
            ),
            effects=(
                ('grasped', 'object'),
            ),
        ),
    ))
    
    goal = (('grasped', 'object'))
    if goal_node_id is not None:
        goal=(
            ('grasped', 'object'),
            ('hand_at', 'l'+str(goal_node_id)),
            ('object_at', 'l'+str(goal_node_id)),
        )
    
    # each node in the graph is a location
    problem = pyddl.Problem(
        domain,
        {
            'location': locations,
        },
        init=[
            ('hand_at', 'l'+str(hand_start_node_id)),
            ('object_at', 'l'+str(object_start_node_id)),
        ] + connections + grasping_locations,
        goal=goal
    )

    plan = pyddl.planner(problem, verbose=verbose)
    if plan is None:
        print('No Plan!')
    else:
        for action in plan:
            print(action)
    
    return plan


def main(**args):
    rospy.init_node('ec_planner')

    # get grasp from graph representation
    grasp_path = None
    while grasp_path is None:
        # get geometry graph
        graph = rospy.wait_for_message('geometry_graph', Graph)
        
        print("Received graph with {} nodes and {} edges.".format(len(graph.nodes), len(graph.edges)))
        
        # identify potential goal nodes (based on grasp type)
        goal_node_ids = [i for i, n in enumerate(graph.nodes) if n.label in args['grasp']]
        
        if len(goal_node_ids) > 0:
            # get all paths that end in goal_ids
            paths = find_all_paths(0, goal_node_ids, graph)
            
            if len(paths) > 0:
                grasp_path = paths[0]
                break
        
        rospy.sleep(0.3)
    
    # Turn grasp into hybrid automaton
    ha = hybrid_automaton_from_motion_sequence(grasp_path)
    
    # call a service
    if args['ros_service_call']:
        call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        call_ha(ha.xml())
    
    # write to file
    if args['file_output']:
        with open('hybrid_automaton.xml', 'w') as outfile:
            outfile.write(ha.xml())
    
    if args['rviz_visualization']:
        pass # TODO
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn path in graph into hybrid automaton.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--ros_service_call', action='store_true', default = False,
                        help='Whether to send the hybrid automaton to a ROS service called /update_hybrid_automaton.')
    parser.add_argument('--file_output', action='store_true', default = False,
                        help='Whether to write the hybrid automaton to a file called hybrid_automaton.xml.')
    grasp_choices = ["any", "edge_grasp", "wall_grasp", "surface_grasp"]
    parser.add_argument('--grasp', choices=grasp_choices, default=grasp_choices[0],
                        help='which grasp type to use')
    parser.add_argument('--rviz_visualization', action='store_true', default = False,
                        help='Whether to send marker messages that can be seen in RViz and represent the chosen grasping motion.')
    args = parser.parse_args()
    
    if args['grasp'] == 'any':
        args['grasp'] = grasp_choices[1:]
    else:
        args['grasp'] = [args['grasp']]
    
    main(**vars(args))
