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
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from subprocess import call
from hybrid_automaton_msgs import srv
from hybrid_automaton_msgs.msg import HAMState

from std_msgs.msg import Header

from pregrasp_msgs.msg import GraspStrategyArray
from pregrasp_msgs.msg import GraspStrategy

from geometry_graph_msgs.msg import Graph

from ec_grasp_planner import srv

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

import pyddl

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')
sys.path.append(pkg_path + '/../hybrid-automaton-tools-py/')
import hatools.components as ha
import hatools.cookbook as cookbook

import handarm_parameters

markers_rviz = MarkerArray()
frames_rviz = []

class GraspPlanner():
    def __init__(self, args):
        # initialize the ros node
        rospy.init_node('ec_planner')
        s = rospy.Service('run_grasp_planner', srv.RunGraspPlanner, lambda msg: self.handle_run_grasp_planner(msg))
        self.tf_listener = tf.TransformListener()
        self.args = args

    # ------------------------------------------------------------------------------------------------
    def handle_run_grasp_planner(self, req):
        self.object_type = req.object_type
        self.grasp_type = req.grasp_type
        self.handarm_params = handarm_parameters.__dict__[req.handarm_type]()

        robot_base_frame = self.args.robot_base_frame
        object_frame = self.args.object_frame
        

        # make sure those frames exist and we can transform between them
        # self.tf_listener.waitForTransform(object_frame, robot_base_frame, rospy.Time(), rospy.Duration(10.0))

        # --------------------------------------------------------
        # Get grasp from graph representation
        if not self.args.bypass:
            print("Using graph")
            grasp_path = None
            while grasp_path is None:
                # Get geometry graph
                graph = rospy.wait_for_message('geometry_graph', Graph)
                graph.header.stamp = rospy.Time.now() + rospy.Duration(0.5)

                # Get the geometry graph frame in robot base frame
                self.tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, graph.header.stamp, rospy.Duration(10.0))
                graph_in_base = self.tf_listener.asMatrix(robot_base_frame, graph.header)

                # Get the object frame in robot base frame
                self.tf_listener.waitForTransform(robot_base_frame, object_frame, graph.header.stamp, rospy.Duration(10.0))
                object_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, rospy.Time(), object_frame))

                print("Received graph with {} nodes and {} edges.".format(len(graph.nodes), len(graph.edges)))

                # Find a path in the ECE graph
                hand_node_id = [n.label for n in graph.nodes].index("Positioning")
                object_node_id = [n.label for n in graph.nodes].index("Slide")

                grasp_path = find_a_path(hand_node_id, object_node_id, graph, self.grasp_type, verbose=True)

                rospy.sleep(0.3)

            # --------------------------------------------------------
            # Turn grasp into hybrid automaton
            ha, self.rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, graph_in_base, object_in_base,
                                                                    self.handarm_params, self.object_type)
        else:
            print("Bypassing graph")
            # Get the object frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, "ifco", rospy.Time.now(), rospy.Duration(1000.0))
            object_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, rospy.Time(), "ifco"))

            self.tf_listener.waitForTransform(robot_base_frame, "ifco", rospy.Time.now(), rospy.Duration(1000.0))
            ifco_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, rospy.Time(), "ifco"))

            self.tf_listener.waitForTransform(robot_base_frame, "wall1", rospy.Time.now(), rospy.Duration(1000.0))
            wall_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, rospy.Time(), "wall1"))

            ha, self.rviz_frames = hybrid_automaton_without_motion_sequence(self.grasp_type, object_in_base, ifco_in_base, wall_in_base,
                                                                    self.handarm_params, self.object_type)
        # --------------------------------------------------------
        # Output the hybrid automaton

        # Call update_hybrid_automaton service to communicate with a hybrid automaton manager (kuka or rswin)
        if self.args.ros_service_call:
            call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
            call_ha(ha.xml())

        # Write to a xml file
        if self.args.file_output:
            with open('hybrid_automaton.xml', 'w') as outfile:
                outfile.write(ha.xml())

        # Publish rviz markers

        if self.args.rviz:
            #print "Press Ctrl-C to stop sending visualization_msgs/MarkerArray on topic '/planned_grasp_path' ..."
            publish_rviz_markers(self.rviz_frames, robot_base_frame, self.handarm_params)
            # rospy.spin()


        return srv.RunGraspPlannerResponse(ha.xml())

# ================================================================================================
def create_surface_grasp_original(object_frame, support_surface_frame, handarm_params, object_type):

    # Get the relevant parameters for hand object combination
    if (object_type in handarm_params['surface_grasp']):            
        params = handarm_params['surface_grasp'][object_type]
    else:
        params = handarm_params['surface_grasp']['object']

    hand_transform = params['hand_transform']
    pregrasp_transform = params['pregrasp_transform']
    post_grasp_transform= params['post_grasp_transform'] # TODO: USE THIS!!!

    drop_off_config = params['drop_off_config']
    downward_force = params['downward_force']
    hand_closing_time = params['hand_closing_duration']
    hand_opening_time = params['hand_opening_duration']
    hand_synergy = params['hand_closing_synergy']
    down_speed = params['down_speed']
    up_speed = params['up_speed']

    # Set the initial pose above the object
    goal_ = np.copy(object_frame) #TODO: this should be support_surface_frame
    goal_[:3,3] = tra.translation_from_matrix(object_frame)
    goal_ =  goal_.dot(hand_transform)

    #the grasp frame is symmetrical - check which side is nicer to reach
    #this is a hacky first version for our WAM
    zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    if goal_[0][0]<0:
        goal_ = goal_.dot(zflip_transform)


    # hand pose above object
    pre_grasp_pose = goal_.dot(pregrasp_transform)

    # Set the directions to use TRIK controller with
    # Down speed is positive because it is defined on the EE frame
    dirDown = tra.translation_matrix([0, 0, -down_speed]);
    # Up speed is also positive because it is defined on the world frame
    dirUp = tra.translation_matrix([0, 0, up_speed]);

    # Set the frames to visualize with RViz
    rviz_frames = []
    rviz_frames.append(object_frame)
    rviz_frames.append(goal_)
    rviz_frames.append(pre_grasp_pose)


    # assemble controller sequence
    control_sequence = []

    # # 1. Go above the object - Pregrasp
    # control_sequence.append(
    #     ha.InterpolatedHTransformControlMode(pre_grasp_pose, controller_name='GoAboveIFCO', goal_is_relative='0',
    #                                          name='Pre_preGrasp'))
    #
    # # 1b. Switch when hand reaches the goal pose
    # control_sequence.append(ha.FramePoseSwitch('Pre_preGrasp', 'Pregrasp', controller='GoAboveIFCO', epsilon='0.01'))

    # 2. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pre_grasp_pose, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'Pregrasp'))
 
    # 2b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Pregrasp', 'GoDown', controller = 'GoAboveObject', epsilon = '0.01'))
 
    # 3. Go down onto the object (relative in world frame) - Godown
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirDown, controller_name='GoDown', goal_is_relative='1', name="GoDown",
                                             reference_frame="world"))

    force  = np.array([0, 0, 0.5*downward_force, 0, 0, 0])
    # 3b. Switch when goal is reached
    control_sequence.append(ha.ForceTorqueSwitch('GoDown', 'softhand_close',  goal = force,
        norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", goal_is_relative = '1', frame_id = 'world', port = '2'))

    # 4. Maintain the position
    desired_displacement = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    force_gradient = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])    

    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_close', synergy = hand_synergy,
                                                        desired_displacement = desired_displacement, 
                                                        force_gradient = force_gradient, 
                                                        desired_force_dimension = desired_force_dimension))
    else:
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = 'softhand_close', synergy = '1'))


    # 4b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration = hand_closing_time))

    # # 5. Rotate hand after closing and before lifting it up
    # # relative to current hand pose
    # control_sequence.append(
    #     ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate', goal_is_relative='1', ))

    # # 5b. Switch when hand rotated
    # control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp', controller='PostGraspRotate', epsilon='0.01', goal_is_relative='1', reference_frame = 'EE'))

    # 6. Lift upwards
    control_sequence.append(ha.InterpolatedHTransformControlMode(dirUp, controller_name = 'GoUpHTransform', name = 'GoUp', goal_is_relative='1', reference_frame="world"))
 
    # 6b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoUp', 'softhand_open', duration = 12))

    # # 7. Go to dropOFF
    # control_sequence.append(ha.JointControlMode(drop_off_config, controller_name = 'GoToDropJointConfig', name = 'GoDropOff'))
 
    # # 7.b  Switch when joint is reached
    # control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'softhand_open', controller = 'GoToDropJointConfig', epsilon = str(math.radians(7.))))

    # 8. Release SKU
    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_open', synergy = hand_synergy,
                                                        desired_displacement = desired_displacement,
                                                        force_gradient = force_gradient,
                                                        desired_force_dimension = desired_force_dimension))
    else:
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = '1'))


    # 8b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'finished', duration = hand_opening_time))

    # 9. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name  = 'finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller( name = 'JointSpaceController', type = 'InterpolatedJointController', goal  = np.zeros(7),
                                   goal_is_relative = 0, v_max = '[0,0]', a_max = '[0,0]'))
    finishedMode.set(finishedSet)  
    control_sequence.append(finishedMode)    
    

    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames

# ================================================================================================
def create_wall_grasp(object_frame, support_surface_frame, wall_frame, handarm_params, object_type):

    # Get the parameters from the handarm_parameters.py file
    if (object_type in handarm_params['wall_grasp']):
        params = handarm_params['wall_grasp'][object_type]
    else:
        params = handarm_params['wall_grasp']['object']

    # initial configuration above IFCO. Should be easy to go from here to pregrasp pose
    # TODO remove this once we have configuration space planning capabilities
    initial_jointConf = params['initial_goal']

    # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
    hand_transform = params['hand_transform']

    # transformation to apply after grasping
    post_grasp_transform = params['post_grasp_transform']


    downward_force = params['table_force']
    sliding_dist = params['sliding_dist']
    up_dist = params['up_dist']
    lift_dist = params['lift_dist']
    down_dist = params['down_dist']
    wall_force = params['wall_force']
    pre_approach_transform = params['pre_approach_transform']
    drop_off_config = params['drop_off_config']
    go_down_velocity = params['go_down_velocity']
    go_up_velocity = params['go_up_velocity']
    slide_velocity = params['slide_velocity']
    hand_closing_duration = params['hand_closing_duration']
    hand_opening_duration = params['hand_opening_duration']

    # Get the pose above the object
    global rviz_frames
    rviz_frames = []

    # this is the EC frame. It is positioned like object and oriented to the wall
    ec_frame = np.copy(wall_frame)
    ec_frame[:3, 3] = tra.translation_from_matrix(object_frame)
    # apply hand transformation
    ec_frame = (ec_frame.dot(hand_transform))

    # This is behind the object (10cm) with the palm facing the wall
    #position_behind_object = ec_frame.dot(tra.translation_matrix([0, 0, -0.1]))


    # the pre-approach pose should be:
    # - floating above and behind the object,
    # - fingers pointing downwards
    # - palm facing the object and wall
    pre_approach_pose = ec_frame.dot(pre_approach_transform)


    # Rviz debug frames
    rviz_frames.append(wall_frame)
    rviz_frames.append(ec_frame)
    #rviz_frames.append(position_behind_object)
    rviz_frames.append(pre_approach_pose)

    # use the default synergy
    hand_synergy = 1

    control_sequence = []

    # 0. initial position above ifco
    control_sequence.append(
        ha.JointControlMode(initial_jointConf, name='InitialJointConfig', controller_name='initialJointCtrl'))

    # 0b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('InitialJointConfig', 'PreGrasp', controller='initialJointCtrl',
                                                        epsilon=str(math.radians(7.))))

    # 1. Go above the object
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(pre_approach_pose, controller_name='GoAboveObject', goal_is_relative='0',
                                             name="PreGrasp"))

    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'GoDown', controller='GoAboveObject', epsilon='0.01'))

    # 2. Go down onto the object/table, in world frame
    dirDown = tra.translation_matrix([0, 0, -go_down_velocity])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirDown, controller_name='GoDown', goal_is_relative='1', name="GoDown",
                                             reference_frame="world"))
    
    # 2b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown', 'LiftHand', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', port='2'))

    # 3. Lift upwards so the hand doesn't slide on table surface
    dirLift = tra.translation_matrix([0, 0, go_up_velocity])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirLift, controller_name='Lift1', goal_is_relative='1', name="LiftHand",
                                             reference_frame="world"))

    # 3b. We switch after a short time as this allows us to do a small, precise lift motion
    # TODO partners: this can be replaced by a frame pose switch if your robot is able to do small motions precisely
    control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=1))

    # 4. Go towards the wall to slide object to wall
    dirWall = tra.translation_matrix([0, 0, slide_velocity])
    #TODO sliding_distance should be computed from wall and hand frame.

    # slide direction is given by the normal of the wall
    # dirWall[:3, 3] = wall_frame[:3, :3].dot(dirWall[:3, 3])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirWall, controller_name='SlideToWall', goal_is_relative='1',
                                             name="SlideToWall", reference_frame="EE"))

    # 4b. Switch when the f/t sensor is triggered with normal force from wall
    # TODO arne: needs tuning
    force = np.array([0, 0, wall_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', 'GoDownAgain', 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame, port='2'))

    # 2. Go down onto the object/table, in world frame
    dirDown = tra.translation_matrix([0, 0, -go_down_velocity])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirDown, controller_name='GoDownAgain', goal_is_relative='1', name="GoDownAgain",
                                             reference_frame="world"))
    
    # 2b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDownAgain', 'softhand_close', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', port='2'))

    # 5. Maintain contact while closing the hand
    if handarm_params['isForceControllerAvailable']:
        # apply force on object while closing the hand
        # TODO arne: validate these values
        desired_displacement = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        force_gradient = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
        desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        control_sequence.append(ha.HandControlMode_ForceHT(name='softhand_close', synergy=hand_synergy,
                                                           desired_displacement=desired_displacement,
                                                           force_gradient=force_gradient,
                                                           desired_force_dimension=desired_force_dimension))
    else:
        # just close the hand
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = 'softhand_close', synergy = '1'))

    # 5b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration=hand_closing_duration))

# #    # 6. Move hand after closing and before lifting it up
# #    # relative to current hand pose
# #    control_sequence.append(
# #        ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate',
# #                                 goal_is_relative='1', ))

# #    # 6b. Switch when hand reaches post grasp pose
# #    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp', controller='PostGraspRotate', epsilon='0.01',
# #                                               goal_is_relative='1', reference_frame='EE'))

    # 7. Lift upwards (+z in world frame)
    dirUp = tra.translation_matrix([0, 0, go_up_velocity])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirUp, controller_name='GoUpHTransform', name='GoUp', goal_is_relative='1',
                                             reference_frame="world"))

#     # 7b. Switch when lifting motion is completed
#     control_sequence.append(
#         ha.FramePoseSwitch('GoUp', 'GoDropOff', controller='GoUpHTransform', epsilon='0.01', goal_is_relative='1',
#                            reference_frame="world"))
    control_sequence.append(ha.TimeSwitch('GoUp', 'softhand_open', duration = 7))
#     # 8. Go to drop off configuration
#     control_sequence.append(
#         ha.JointControlMode(drop_off_config, controller_name='GoToDropJointConfig', name='GoDropOff'))

#     # 8.b  Switch when configuration is reached
#     control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'softhand_open', controller='GoToDropJointConfig',
#                                                         epsilon=str(math.radians(7.))))

    # 9. Release SKU
    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_open', synergy = hand_synergy,
                                                        desired_displacement = desired_displacement,
                                                        force_gradient = force_gradient,
                                                        desired_force_dimension = desired_force_dimension))
    else:
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = '1'))


    # 9b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'finished', duration = hand_opening_duration))

    # 10. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name='finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller(name='JointSpaceController', type='InterpolatedJointController', goal=np.zeros(7),
                                  goal_is_relative=0, v_max='[0,0]', a_max='[0,0]'))

    finishedMode.set(finishedSet)
    control_sequence.append(finishedMode)

    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames

# ================================================================================================
def transform_msg_to_homogenous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]), tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))

# ================================================================================================
def homogenous_tf_to_pose_msg(htf):
    return Pose(position = Point(*tra.translation_from_matrix(htf).tolist()), orientation = Quaternion(*tra.quaternion_from_matrix(htf).tolist()))

# ================================================================================================
def get_node_from_actions(actions, action_name, graph):
    return graph.nodes[[int(m.sig[1][1:]) for m in actions if m.name == action_name][0]]

# ================================================================================================
def hybrid_automaton_from_motion_sequence(motion_sequence, graph, T_robot_base_frame, T_object_in_base, handarm_params, object_type):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))

    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label
    #grasp_frame = grasp_frames[grasp_type]

    print("Creating hybrid automaton for object {} and grasp type {}.".format(object_type, grasp_type))
    if grasp_type == 'EdgeGrasp':
        raise "Edge grasp is not supported yet"
        #support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        #support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        #edge_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        #edge_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(edge_frame_node.transform))
        return create_edge_grasp(T_object_in_base, support_surface_frame, edge_frame, handarm_params)
    elif grasp_type == 'WallGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        wall_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        wall_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(wall_frame_node.transform))
        return create_wall_grasp(T_object_in_base, support_surface_frame, wall_frame, handarm_params, object_type)
    elif grasp_type == 'SurfaceGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        return create_surface_grasp(T_object_in_base, support_surface_frame, handarm_params, object_type)
    else:
        raise "Unknown grasp type: ", grasp_type

# ================================================================================================
def hybrid_automaton_without_motion_sequence(grasp_type, T_object_in_base, T_ifco_in_base, T_wall_in_base, handarm_params, object_type):

    print("Creating hybrid automaton for object {} and grasp type {}.".format(object_type, grasp_type))
    if grasp_type == '-WallGrasp':
        return create_wall_grasp(T_object_in_base, T_ifco_in_base, T_wall_in_base, handarm_params, object_type)
    elif grasp_type == '-SurfaceGrasp':
        return create_surface_grasp(T_object_in_base, T_ifco_in_base, handarm_params, object_type)
    else:
        raise "Unknown grasp type: ", grasp_type

# ================================================================================================
def find_a_path(hand_start_node_id, object_start_node_id, graph, goal_node_labels, verbose = False):
    locations = ['l'+str(i) for i in range(len(graph.nodes))]

    connections = [('connected', 'l'+str(e.node_id_start), 'l'+str(e.node_id_end)) for e in graph.edges]
    grasping_locations = [('is_grasping_location', 'l'+str(i)) for i, n in enumerate(graph.nodes) if n.label in goal_node_labels or n.label+'_'+str(i) in goal_node_labels]

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
        goal=(
            ('grasped', 'object'),
        )
    )

    plan = pyddl.planner(problem, verbose=verbose)
    if plan is None:
        print('No Plan!')
    else:
        for action in plan:
            print(action)

    return plan

# ================================================================================================
def publish_rviz_markers(frames, frame_id, handarm_params):

    timestamp = rospy.Time.now()

    global markers_rviz
    global frames_rviz

    markers_rviz = MarkerArray()
    for i, f in enumerate(frames):
        msg = Marker()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.frame_locked = True # False
        msg.id = i
        msg.type = Marker.MESH_RESOURCE
        msg.action = Marker.ADD
        msg.lifetime = rospy.Duration()
        msg.color.r = msg.color.g = msg.color.b = msg.color.a = 0
        msg.mesh_use_embedded_materials = True
        msg.mesh_resource = handarm_params["mesh_file"]
        msg.scale.x = msg.scale.y = msg.scale.z = handarm_params["mesh_file_scale"]
        #msg.mesh_resource = mesh_resource
        msg.pose = homogenous_tf_to_pose_msg(f)

        markers_rviz.markers.append(msg)

    for f1, f2 in zip(frames, frames[1:]):
        msg = Marker()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.frame_locked = True # False
        msg.id = markers_rviz.markers[-1].id + 1
        msg.action = Marker.ADD
        msg.lifetime = rospy.Duration()
        msg.type = Marker.ARROW
        msg.color.g = msg.color.b = 0
        msg.color.r = msg.color.a = 1
        msg.scale.x = 0.01 # shaft diameter
        msg.scale.y = 0.03 # head diameter
        msg.points.append(homogenous_tf_to_pose_msg(f1).position)
        msg.points.append(homogenous_tf_to_pose_msg(f2).position)

        markers_rviz.markers.append(msg)
   
    frames_rviz = frames
# ================================================================================================

# ================================================================================================
def create_surface_grasp(object_frame, support_surface_frame, handarm_params, object_type):

    #internal experiments parameter
    hand_orientation = 0
    hand_object_distance = 0.15
    approach_direction = 45

    # Get the relevant parameters for hand object combination
    if (object_type in handarm_params['surface_grasp']):
        params = handarm_params['surface_grasp'][object_type]
    else:
        params = handarm_params['surface_grasp']['object']

    hand_transform = params['hand_transform']
    pregrasp_transform = params['pregrasp_transform']
    #post_grasp_transform= params['post_grasp_transform'] # TODO: USE THIS!!!   

    #drop_off_config = params['drop_off_config']
    downward_force = params['downward_force']
    hand_closing_time = params['hand_closing_duration']
    hand_opening_time = params['hand_opening_duration']
    hand_synergy = params['hand_closing_synergy']
    down_speed = params['down_speed']
    up_speed = params['up_speed']
    
    approach_direction_vector = np.dot(tra.rotation_matrix(math.radians(-approach_direction),[1, 0 , 0]), np.array([0,down_speed,0,0]))[0:3]
    print(approach_direction_vector)
    print("hand_transform")
    print(hand_transform)
    ad_transform = tra.translation_matrix(approach_direction_vector)


    # Set the initial pose above the object
    goal_ = np.copy(object_frame) #TODO: this should be support_surface_frame
    hand_init_pose = tra.concatenate_matrices(tra.translation_matrix([0, hand_object_distance, 0]), tra.rotation_matrix(math.radians(180.), [0, 1, 0]), tra.rotation_matrix(math.radians(90.), [1, 0, 0]) ) #TODO change from 

    goal_ = goal_.dot(hand_init_pose) 

    goal_ = goal_.dot(tra.rotation_matrix(math.radians(-approach_direction),[1, 0 , 0], [0, 0 , hand_object_distance]))

    goal_[0,3] = goal_[0,3] - 0.05 #cheat for testing purpose
    goal_ = goal_.dot(tra.rotation_matrix(math.radians(hand_orientation),[0, 0 , 1]))
    # print("goal_1")
    # print(goal_)
    # #goal_[:3,3] = tra.translation_from_matrix(object_frame)
    # #print("goal_2")
    # #print(goal_)
    # #goal_ = goal_.dot(hand_transform)
    # print("goal_3")
    # print(goal_)    
    #the grasp frame is symmetrical - check which side is nicer to reach
    #this is a hacky first version for our WAM
    # zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    # if goal_[0][0]<0:
    #     goal_ = goal_.dot(zflip_transform)

    # print("goal_4")
    # print(goal_)
    #pregrasp_transform = pregrasp_transform.dot(tra.rotation_matrix(math.radians(hand_orientation), [0, 0, 1]))
    # hand pose above object
    #pre_grasp_pose = goal_.dot(pregrasp_transform)

    pre_grasp_pose = np.copy(goal_)

    # pose above the placement location
    pre_placement_pose = pre_grasp_pose.dot(tra.rotation_matrix(math.radians(-90.0), [0, 0, 1],[0, 0, 0,0]))
    pre_placement_pose[0,3] = pre_grasp_pose[1,3]
    pre_placement_pose[1,3] = pre_grasp_pose[0,3]

    placement_pose = np.copy(pre_placement_pose)
    placement_pose[2,3] = placement_pose[2,3] - 0.1


    # Set the directions to use TRIK controller with
    # Down speed is positive because it is defined on the EE frame
    dirDown = tra.translation_matrix([0, 0, -down_speed]);
    # Up speed is also positive because it is defined on the world frame
    dirUp = tra.translation_matrix([0, 0, up_speed]);

    # Set the frames to visualize with RViz
    rviz_frames = []
    rviz_frames.append(object_frame)
    rviz_frames.append(goal_)
    rviz_frames.append(pre_grasp_pose)


    # assemble controller sequence
    control_sequence = []

    # # 1. Go above the object - Pregrasp
    # control_sequence.append(
    #     ha.InterpolatedHTransformControlMode(pre_grasp_pose, controller_name='GoAboveIFCO', goal_is_relative='0',
    #                                          name='Pre_preGrasp'))
    #
    # # 1b. Switch when hand reaches the goal pose
    # control_sequence.append(ha.FramePoseSwitch('Pre_preGrasp', 'Pregrasp', controller='GoAboveIFCO', epsilon='0.01'))

    # 2. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pre_grasp_pose, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'Pregrasp'))
 
    # 2b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Pregrasp', 'GoDown', controller = 'GoAboveObject', epsilon = '0.01'))
 
    # 3. Go down onto the object (relative in world frame) - Godown
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(ad_transform, controller_name='GoDown', goal_is_relative='1', name="GoDown",
                                             reference_frame="world"))

    force  = np.array([0, 0, 0.5*downward_force, 0, 0, 0])
    # 3b. Switch when goal is reached
    control_sequence.append(ha.ForceTorqueSwitch('GoDown', 'softhand_close',  goal = force,
        norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", goal_is_relative = '1', frame_id = 'EE', port = '2'))

    # 4. Maintain the position
    desired_displacement = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    force_gradient = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])    

    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_close', synergy = hand_synergy,
                                                        desired_displacement = desired_displacement, 
                                                        force_gradient = force_gradient, 
                                                        desired_force_dimension = desired_force_dimension))
    else:
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = 'softhand_close', synergy = '1'))


    # 4b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration = hand_closing_time))

    # # 5. Rotate hand after closing and before lifting it up
    # # relative to current hand pose
    # control_sequence.append(
    #     ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate', goal_is_relative='1', ))

    # # 5b. Switch when hand rotated
    # control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp', controller='PostGraspRotate', epsilon='0.01', goal_is_relative='1', reference_frame = 'EE'))

    # 6. Lift upwards
    control_sequence.append(ha.InterpolatedHTransformControlMode(dirUp, controller_name = 'GoUpHTransform', name = 'GoUp', goal_is_relative='1', reference_frame="world"))
 
    # 6b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoUp', 'Preplacement', duration = 8))

    #  # 2. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pre_placement_pose, controller_name = 'GoAbovePlacement', goal_is_relative='0', name = 'Preplacement'))
 
    # # 2b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Preplacement', 'GoDown2', controller = 'GoAbovePlacement', epsilon = '0.01'))

    # # # 7. Go to dropOFF
    # control_sequence.append(ha.JointControlMode(drop_off_config, controller_name = 'GoToDropJointConfig', name = 'GoDropOff'))
 
    # # # 7.b  Switch when joint is reached
    # control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'softhand_open', controller = 'GoToDropJointConfig', epsilon = str(math.radians(7.))))

    # 7. Go Down upwards
    #control_sequence.append(ha.InterpolatedHTransformControlMode(dirDown, controller_name = 'GoToDropOff', name = 'GoDown2', goal_is_relative='1', reference_frame="world"))
 
    # 7b. Switch after a certain amount of time
    #control_sequence.append(ha.TimeSwitch('GoDown2', 'softhand_open', duration = 3))

    control_sequence.append(ha.InterpolatedHTransformControlMode(placement_pose, controller_name = 'GoToDropOff', goal_is_relative='0', name = 'GoDown2'))
 
    # # 2b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('GoDown2', 'softhand_open', controller = 'GoToDropOff', epsilon = '0.01'))


    # 8. Release SKU
    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_open', synergy = hand_synergy,
                                                        desired_displacement = desired_displacement,
                                                        force_gradient = force_gradient,
                                                        desired_force_dimension = desired_force_dimension))
    else:
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = '1'))


    # 8b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'finished', duration = hand_opening_time))

    # 9. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name  = 'finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller( name = 'JointSpaceController', type = 'InterpolatedJointController', goal  = np.zeros(7),
                                   goal_is_relative = 0, v_max = '[0,0]', a_max = '[0,0]'))
    finishedMode.set(finishedSet)  
    control_sequence.append(finishedMode)    
    

    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames

if __name__ == '__main__':

    print(sys.argv)
    parser = argparse.ArgumentParser(description='Turn path in graph into hybrid automaton.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ros_service_call', action='store_true', default = False,
                        help='Whether to send the hybrid automaton to a ROS service called /update_hybrid_automaton.')
    parser.add_argument('--file_output', action='store_true', default = False,
                        help='Whether to write the hybrid automaton to a file called hybrid_automaton.xml.')
    #grasp_choices = ["any", "EdgeGrasp", "WallGrasp", "SurfaceGrasp"]
    grasp_choices = ["any", "WallGrasp", "SurfaceGrasp"]
    # parser.add_argument('--grasp', choices=grasp_choices, default=grasp_choices[0],
    #                     help='Which grasp type to use.')
    # parser.add_argument('--grasp_id', type=int, default=-1,
    #                    help='Which specific grasp to use. Ignores any values < 0.')
    parser.add_argument('--rviz', action='store_true', default = False,
                        help='Whether to send marker messages that can be seen in RViz and represent the chosen grasping motion.')
    parser.add_argument('--robot_base_frame', type=str, default = 'base_link',
                        help='Name of the robot base frame.')
    parser.add_argument('--object_frame', type=str, default = 'object',
                        help='Name of the object frame.')
    parser.add_argument('--bypass', action='store_true', default = False,
                        help='Whether to bypass graph.')
    # parser.add_argument('--handarm', type=str, default = 'RBOHand2WAM',
    #                     help='Python class that contains configuration parameters for hand and arm-specific properties.')


    # args = parser.parse_args()
    args = parser.parse_args(rospy.myargv()[1:])

    # if args.grasp == 'any':
    #     args.grasp = grasp_choices[1:]
    # else:
    #     args.grasp = [args.grasp]

    # if args.grasp_id >= 0:
    #     tmp = [g + '_' + str(args.grasp_id) for g in args.grasp]
    #     args.grasp = tmp

    robot_base_frame = args.robot_base_frame

    planner = GraspPlanner(args)



    r = rospy.Rate(5);

    marker_pub = rospy.Publisher('planned_grasp_path', MarkerArray, queue_size=1, latch=False)
    br = tf.TransformBroadcaster()

    global frames_rviz, markers_rviz
    while not rospy.is_shutdown():
        marker_pub.publish(markers_rviz)
        for i, f in enumerate(frames_rviz):
            br.sendTransform(tra.translation_from_matrix(f),
                             tra.quaternion_from_matrix(f),
                             rospy.Time.now(),
                             "dgb_frame_" + str(i),
                             robot_base_frame)

        r.sleep()

# ================================================================================================
# ================================================================================================
