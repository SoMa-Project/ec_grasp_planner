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

from random import SystemRandom

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
from hybrid_automaton_msgs import srv as ha_srv
from hybrid_automaton_msgs.msg import HAMState

from std_msgs.msg import Header

from pregrasp_msgs.msg import GraspStrategyArray
from pregrasp_msgs.msg import GraspStrategy

from geometry_graph_msgs.msg import Graph

from ec_grasp_planner import srv as plan_srv

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

from pregrasp_msgs import srv as vision_srv

from xper_data import srv as xper_srv

import pyddl

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')
sys.path.append(pkg_path + '/../hybrid-automaton-tools-py/')
import hatools.components as ha
import hatools.cookbook as cookbook
import tf_conversions.posemath as pm

import handarm_parameters

markers_rviz = MarkerArray()
frames_rviz = []

class GraspPlanner():
    def __init__(self, args):
        # initialize the ros node
        rospy.init_node('ec_planner')
        s = rospy.Service('run_grasp_planner', plan_srv.RunGraspPlanner, lambda msg: self.handle_run_grasp_planner(msg))
        self.tf_listener = tf.TransformListener()
        self.args = args

    # ------------------------------------------------------------------------------------------------
    def handle_run_grasp_planner(self, req):
        
        print('Handling grasp planner service call')
        self.object_type = req.object_type

        #todo: more failure handling here for bad service parameters

        self.handarm_params = handarm_parameters.__dict__[req.handarm_type]()

        # Get the relevant parameters for hand object combination

        rospy.wait_for_service('compute_ec_graph')

        try:
            call_vision = rospy.ServiceProxy('compute_ec_graph', vision_srv.ComputeECGraph)
            res = call_vision(self.object_type)
            graph = res.graph
            objects = res.objects.objects
            print("Objects found: " + str(len(objects)))
        except rospy.ServiceException, e:
            raise rospy.ServiceException("Vision service call failed: %s" % e)
            return plan_srv.RunGraspPlannerResponse("")

        robot_base_frame = self.args.robot_base_frame

        object_id = SystemRandom().randrange(0,len(objects))
        
        object_frame = objects[object_id].transform

        time = rospy.Time(0)
        graph.header.stamp = time
        object_frame.header.stamp = time
        bounding_box = objects[object_id].boundingbox

        self.tf_listener.waitForTransform(robot_base_frame, "/ifco", time, rospy.Duration(2.0))
        ifco_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, time, "ifco"))

        # --------------------------------------------------------
        # Get grasp from graph representation
        grasp_path = None
        while grasp_path is None:
            # Get the geometry graph frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, time, rospy.Duration(2.0))
            graph_in_base = self.tf_listener.asMatrix(robot_base_frame, graph.header)


            # Get the object frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, object_frame.header.frame_id, time, rospy.Duration(2.0))
            camera_in_base = self.tf_listener.asMatrix(robot_base_frame, object_frame.header)
            object_in_camera = pm.toMatrix(pm.fromMsg(object_frame.pose))
            object_in_base = camera_in_base.dot(object_in_camera)

            
            #get grasp type
            self.grasp_type = req.grasp_type
            if self.grasp_type == 'UseHeuristics':
                if self.object_type in self.handarm_params:
                    obj_bbox_uncertainty_offset = self.handarm_params[self.object_type]['obj_bbox_uncertainty_offset']
                else:
                    obj_bbox_uncertainty_offset = self.handarm_params['object']['obj_bbox_uncertainty_offset']
                self.grasp_type, wall_id = grasp_heuristics(ifco_in_base, object_in_base, bounding_box, obj_bbox_uncertainty_offset)
                print("GRASP HEURISTICS " + self.grasp_type + " " + wall_id)

                # call_xper = rospy.ServiceProxy('pregrasp_pose', xper_srv.ProvidePreGraspPose)
                # res = call_xper(pm.toMsg(pm.fromMatrix(ifco_in_base)), pm.toMsg(pm.fromMatrix(object_in_base)))
                # print("REACHABILITY & EXPERIMENTS node proposes a " + res.grasp_type + " grasp")
                # print("approach_direction: " + str(res.approach_direction))
                # print("hand_orientation: " + str(res.hand_orientation))
                # print("plane_orientation: " + str(res.plane_orientation))
            else:                
                wall_id = "wall1"
                grasp_choices = ["any", "WallGrasp", "SurfaceGrasp"]
                if self.grasp_type not in grasp_choices:
                    raise rospy.ServiceException("grasp_type not supported. Choose from [any,WallGrasp,SurfaceGrasp]")
                    return

            print("Received graph with {} nodes and {} edges.".format(len(graph.nodes), len(graph.edges)))

            # Find a path in the ECE graph
            hand_node_id = [n.label for n in graph.nodes].index("Positioning")
            object_node_id = [n.label for n in graph.nodes].index("Slide")

            grasp_path = find_a_path(hand_node_id, object_node_id, graph, self.grasp_type, verbose=True)

            rospy.sleep(0.3)


        

        # --------------------------------------------------------
        # Turn grasp into hybrid automaton
        ha, self.rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, graph_in_base, object_in_base, bounding_box,
                                                                self.handarm_params, self.object_type, wall_id, ifco_in_base)
                                                
        # --------------------------------------------------------
        # Output the hybrid automaton

        print("generated grasping ha")

        # Call update_hybrid_automaton service to communicate with a hybrid automaton manager (kuka or rswin)
        if self.args.ros_service_call:
            call_ha = rospy.ServiceProxy('update_hybrid_automaton', ha_srv.UpdateHybridAutomaton)
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


        return plan_srv.RunGraspPlannerResponse(ha.xml())

# ================================================================================================
def getParam(obj_type_params, obj_params, paramKey):
    param = obj_type_params.get(paramKey)
    if param is None:
        param = obj_params.get(paramKey)
    return param


# ================================================================================================
def create_surface_grasp(object_frame, bounding_box, support_surface_frame, handarm_params, object_type):

    # Get the parameters from the handarm_parameters.py file
    obj_type_params = {}
    obj_params = {}
    if (object_type in handarm_params['surface_grasp']):            
        obj_type_params = handarm_params['surface_grasp'][object_type]
    if 'object' in handarm_params['surface_grasp']:
        obj_params = handarm_params['surface_grasp']['object']

    hand_transform = getParam(obj_type_params, obj_params, 'hand_transform')
    downward_force = getParam(obj_type_params, obj_params, 'downward_force')
    ee_in_goal_frame = getParam(obj_type_params, obj_params, 'ee_in_goal_frame')

    lift_time = handarm_params['lift_duration']
    place_time = handarm_params['place_duration']    
    down_IFCO_speed = handarm_params['down_IFCO_speed']
    up_IFCO_speed = handarm_params['up_IFCO_speed']
    down_tote_speed = handarm_params['down_tote_speed']
    

    zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    if object_frame[0][1]<0:
        object_frame = object_frame.dot(zflip_transform)

    # Set the initial pose above the object
    goal_ = np.copy(object_frame)
    goal_ = goal_.dot(hand_transform) #this is the pre-grasp transform of the signature frame expressed in the world
    goal_ = goal_.dot(ee_in_goal_frame)

    

    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the EE frame
    down_IFCO_twist = tra.translation_matrix([0, 0, down_IFCO_speed]);
    # Up speed is also positive because it is defined on the world frame
    up_IFCO_twist = tra.translation_matrix([0, 0, up_IFCO_speed]);
    # Down speed is negative because it is defined on the world frame
    down_tote_twist = tra.translation_matrix([0, 0, -down_tote_speed]);

    # Set the frames to visualize with RViz
    rviz_frames = []
    rviz_frames.append(object_frame)
    rviz_frames.append(goal_)

    # assemble controller sequence
    control_sequence = []

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(goal_, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'Pregrasp'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Pregrasp', 'GoDown', controller = 'GoAboveObject', epsilon = '0.01'))
 
    # 2. Go down onto the object (relative in EE frame) - Godown
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(down_IFCO_twist,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="EE",
                                             v_max=down_IFCO_speed))

    # 2b. Switch when force-torque sensor is triggered
    force  = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'softhand_close',
                                                 goal = force,
                                                 norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion = "THRESH_UPPER_BOUND",
                                                 goal_is_relative = '1',
                                                 frame_id = 'world',
                                                 port = '2'))

    desired_displacement = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    force_gradient = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) 
    # 3. Maintain the position
    if handarm_params['isForceControllerAvailable']:   
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_close', synergy = handarm_params['hand_closing_synergy'],
                                                        desired_displacement = desired_displacement, 
                                                        force_gradient = force_gradient, 
                                                        desired_force_dimension = desired_force_dimension))
    elif handarm_params['isInPositionControl']:
        # if hand is controlled in position mode, then call general hand controller
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = 'softhand_close', synergy = '1'))
    else:
        kp = getParam(obj_type_params, obj_params, 'kp')
        # if hand is controlled in current mode, then call IIT's controller
        control_sequence.append(ha.ros_PisaIIThandControlMode(goal = np.array([1.0]), kp=np.array([kp]), hand_max_aperture = handarm_params['hand_max_aperture'], name  = 'softhand_close', 
            bounding_box=np.array([bounding_box.x, bounding_box.y, bounding_box.z]), object_weight=np.array([0.4]), object_type='object', object_pose=object_frame))

    # 3b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration = handarm_params['hand_closing_duration']))

    # 4. Lift upwards
    control_sequence.append(ha.InterpolatedHTransformControlMode(up_IFCO_twist, controller_name = 'GoUpHTransform', name = 'GoUp', goal_is_relative='1', reference_frame="world"))
 
    # 4b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoUp', 'Preplacement', duration = lift_time))

    # 5. Go to Preplacement
    control_sequence.append(ha.InterpolatedHTransformControlMode(handarm_params['pre_placement_pose'], controller_name = 'GoAbovePlacement', goal_is_relative='0', name = 'Preplacement'))
   
    # 5b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Preplacement', 'GoDown2', controller = 'GoAbovePlacement', epsilon = '0.01'))

    # 6. Go Down
    control_sequence.append(ha.InterpolatedHTransformControlMode(down_tote_twist, controller_name = 'GoToDropOff', name = 'GoDown2', goal_is_relative='1', reference_frame="world"))
 
    # 6b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoDown2', 'softhand_open', duration = place_time))

    # 7. Release SKU
    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_open', synergy = handarm_params['hand_closing_synergy'],
                                                        desired_displacement = desired_displacement,
                                                        force_gradient = force_gradient,
                                                        desired_force_dimension = desired_force_dimension))
    else:
        # if hand is controlled in position mode, then call general hand controller
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = handarm_params['hand_closing_synergy']))

    # 7b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'finished', duration = handarm_params['hand_opening_duration']))

    # 8. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name  = 'finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller( name = 'JointSpaceController', type = 'InterpolatedJointController', goal  = np.zeros(7),
                                   goal_is_relative = 1, v_max = '[0,0]', a_max = '[0,0]'))
    finishedMode.set(finishedSet)  
    control_sequence.append(finishedMode)    
    
    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames


# ================================================================================================
def create_wall_grasp(object_frame, bounding_box, support_surface_frame, wall_frame, handarm_params, object_type):

    # Get the parameters from the handarm_parameters.py file
    obj_type_params = {}
    obj_params = {}
    if object_type in handarm_params['wall_grasp']:            
        obj_type_params = handarm_params['wall_grasp'][object_type]
    if 'object' in handarm_params['wall_grasp']:
        obj_params = handarm_params['wall_grasp']['object']

    hand_transform = getParam(obj_type_params, obj_params, 'hand_transform')
    downward_force = getParam(obj_type_params, obj_params, 'downward_force')
    wall_force = getParam(obj_type_params, obj_params, 'wall_force')
    slide_IFCO_speed = getParam(obj_type_params, obj_params, 'slide_speed')
    pre_approach_transform = getParam(obj_type_params, obj_params, 'pre_approach_transform')
    scooping_angle_deg = getParam(obj_type_params, obj_params, 'scooping_angle_deg')
    move_up_after_contact_goal = getParam(obj_type_params, obj_params, 'move_up_after_contact_goal')
    
    vision_params = {}
    if object_type in handarm_params:
        vision_params = handarm_params[object_type]
    offset = getParam(vision_params, handarm_params['object'], 'obj_bbox_uncertainty_offset')
    if abs(object_frame[:3,0].dot(wall_frame[:3,0])) > abs(object_frame[:3,1].dot(wall_frame[:3,0])):
        pre_approach_transform[2,3] = pre_approach_transform[2,3] - bounding_box.y/2 - offset 
    else:
        pre_approach_transform[2,3] = pre_approach_transform[2,3] - bounding_box.x/2 - offset

    post_grasp_transform = getParam(obj_type_params, obj_params, 'post_grasp_transform')

    rotate_time = handarm_params['rotate_duration']
    lift_time = handarm_params['lift_duration']
    place_time = handarm_params['place_duration']    
    down_IFCO_speed = handarm_params['down_IFCO_speed']
    up_IFCO_speed = handarm_params['up_IFCO_speed']
    up_IFCO_speed_slow = handarm_params['up_IFCO_speed_slow']
    down_tote_speed = handarm_params['down_IFCO_speed']


    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the world frame
    down_IFCO_twist = tra.translation_matrix([0, 0, -down_IFCO_speed]);
    # Up speed is also positive because it is defined on the world frame
    up_IFCO_twist = tra.translation_matrix([0, 0, up_IFCO_speed]);
    # Slow Up speed is also positive because it is defined on the world frame
    up_IFCO_twist_slow = tra.translation_matrix([0, 0, up_IFCO_speed_slow]);
    
    # Down speed is negative because it is defined on the world frame
    down_tote_twist = tra.translation_matrix([0, 0, -down_tote_speed]);
    # Slide speed is positive because it is defined on the EE frame + rotation of the scooping angle    
    slide_IFCO_twist = tra.rotation_matrix(math.radians(-scooping_angle_deg), [0, 1, 0]).dot(tra.translation_matrix([0, 0, slide_IFCO_speed]));
    slide_IFCO_twist =  tra.translation_matrix(tra.translation_from_matrix(slide_IFCO_twist))

    rviz_frames = []

    # this is the EC frame. It is positioned like object and oriented to the wall
    ec_frame = np.copy(wall_frame)
    ec_frame[:3, 3] = tra.translation_from_matrix(object_frame)
    ec_frame = ec_frame.dot(hand_transform)

    pre_approach_pose = ec_frame.dot(pre_approach_transform)

    # Rviz debug frames
    rviz_frames.append(object_frame)
    rviz_frames.append(wall_frame)
    rviz_frames.append(ec_frame)
    rviz_frames.append(pre_approach_pose)

    control_sequence = []

    # 1. Go above the object
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(pre_approach_pose, controller_name='GoAboveObject', goal_is_relative='0',
                                             name="PreGrasp"))

    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'GoDown', controller='GoAboveObject', epsilon='0.01'))

    # 2. Go down onto the object/table, in world frame
    control_sequence.append( ha.InterpolatedHTransformControlMode(down_IFCO_twist,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="world"))

    # 2b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'LiftHand',
                                                 goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))


    # 3. Lift upwards so the hand doesn't slide on table surface
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(up_IFCO_twist_slow, controller_name='Lift1', goal_is_relative='1', name="LiftHand",
                                             reference_frame="world"))

    # 3b. We switch after a short time as this allows us to do a small, precise lift motion
    control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=5))
    control_sequence.append(ha.FramePoseSwitch('LiftHand', 'SlideToWall', goal_is_relative='1', goal=move_up_after_contact_goal, epsilon='0.004', reference_frame="world"))
    # 4. Go towards the wall to slide object to wall
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(slide_IFCO_twist, controller_name='SlideToWall', goal_is_relative='1',
                                             name="SlideToWall", reference_frame="EE"))

    # 4b. Switch when the f/t sensor is triggered with normal force from wall
    force = np.array([0, 0, wall_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', 'softhand_close', 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame, port='2'))
                                                 
    # 5. Maintain contact while closing the hand
    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_close', synergy = handarm_params['hand_closing_synergy'],
                                                        desired_displacement = desired_displacement, 
                                                        force_gradient = force_gradient, 
                                                        desired_force_dimension = desired_force_dimension))
    elif handarm_params['isInPositionControl']:
        # if hand is controlled in position mode, then call general hand controller
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = 'softhand_close', synergy = handarm_params['hand_closing_synergy']))
    else:
        # if hand is controlled in current mode, then call IIT's controller
        kp = getParam(obj_type_params, obj_params, 'kp')
        # if hand is controlled in current mode, then call IIT's controller
        control_sequence.append(ha.ros_PisaIIThandControlMode(goal = np.array([1.0]), kp=np.array([kp]), hand_max_aperture = handarm_params['hand_max_aperture'], name  = 'softhand_close', 
            bounding_box=np.array([bounding_box.x, bounding_box.y, bounding_box.z]), object_weight=np.array([0.4]), object_type='object', object_pose=object_frame))


    # 5b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration=handarm_params['hand_closing_duration']))

    # 6. Rotate hand after closing and before lifting it up relative to current hand pose
    # control_sequence.append(
    #     ha.InterpolatedHTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate', goal_is_relative='1', reference_frame='EE'))

    # 6b. Switch when hand rotated
    # control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp', controller='PostGraspRotate', epsilon='0.01', goal_is_relative='1', reference_frame = 'EE'))
    # control_sequence.append(ha.TimeSwitch('PostGraspRotate', 'GoUp', duration = rotate_time))


    # 7. Lift upwards (+z in world frame)
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(up_IFCO_twist, controller_name='GoUpHTransform', name='GoUp', goal_is_relative='1',
                                             reference_frame="world"))

    # 7b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoUp', 'Preplacement', duration = lift_time))

   
    # 8. Go to Preplacement
    control_sequence.append(ha.InterpolatedHTransformControlMode(handarm_params['pre_placement_pose'], controller_name = 'GoAbovePlacement', goal_is_relative='0', name = 'Preplacement'))
   
    # 8b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Preplacement', 'GoDown2', controller = 'GoAbovePlacement', epsilon = '0.01'))

    # 9. Go Down
    control_sequence.append(ha.InterpolatedHTransformControlMode(down_tote_twist, controller_name = 'GoToDropOff', name = 'GoDown2', goal_is_relative='1', reference_frame="world"))
 
    # 9b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoDown2', 'softhand_open', duration = place_time))

    # 10. Release SKU
    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_open', synergy = handarm_params['hand_closing_synergy'],
                                                        desired_displacement = desired_displacement,
                                                        force_gradient = force_gradient,
                                                        desired_force_dimension = desired_force_dimension))
    else:
        # if hand is controlled in position mode, then call general hand controller
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = handarm_params['hand_closing_synergy']))

    # 10b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'finished', duration = handarm_params['hand_opening_duration']))

    # 11. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name  = 'finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller( name = 'JointSpaceController', type = 'InterpolatedJointController', goal  = np.zeros(7),
                                   goal_is_relative = 1, v_max = '[0,0]', a_max = '[0,0]'))

    finishedMode.set(finishedSet)
    control_sequence.append(finishedMode)

    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames

# ================================================================================================
def transform_msg_to_homogenous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]), 
        tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))

# ================================================================================================
def homogenous_tf_to_pose_msg(htf):
    return Pose(position = Point(*tra.translation_from_matrix(htf).tolist()), orientation = Quaternion(*tra.quaternion_from_matrix(htf).tolist()))

# ================================================================================================
def get_node_from_actions(actions, action_name, graph):
    return graph.nodes[[int(m.sig[1][1:]) for m in actions if m.name == action_name][0]]

# ================================================================================================
def hybrid_automaton_from_motion_sequence(motion_sequence, graph, T_robot_base_frame, T_object_in_base, bounding_box, handarm_params, object_type, wall_id, ifco_in_base):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))

    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label
    #grasp_frame = grasp_frames[grasp_type]
    support_surface_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
    support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))       
       
    print("Creating hybrid automaton for object {} and grasp type {}.".format(object_type, grasp_type))
    if grasp_type == 'EdgeGrasp':
        raise "Edge grasp is not supported yet"
        #support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        #support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        #edge_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        #edge_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(edge_frame_node.transform))
        return create_edge_grasp(T_object_in_base, support_surface_frame, edge_frame, handarm_params)
    elif grasp_type == 'WallGrasp':
        #support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        #support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        # wall_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        # wall1_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(wall_frame_node.transform))
        wall_frame = get_wall_tf(ifco_in_base, wall_id)
        return create_wall_grasp(T_object_in_base, bounding_box, support_surface_frame, wall_frame, handarm_params, object_type)
    elif grasp_type == 'SurfaceGrasp':
        #support_surface_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        #support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        return create_surface_grasp(T_object_in_base, bounding_box, support_surface_frame, handarm_params, object_type)
    else:
        raise "Unknown grasp type: ", grasp_type

# ================================================================================================
def find_a_path(hand_start_node_id, object_start_node_id, graph, goal_node_labels, verbose = False):
    locations = ['l'+str(i) for i in range(len(graph.nodes))]

    # connections = [('connected', 'l'+str(e.node_id_start), 'l'+str(e.node_id_end)) for e in graph.edges]

    connections = [('connected', 'l0', 'l1')] + [('connected', 'l0', 'l'+str(i)) for i, n in enumerate(graph.nodes) if n.label in goal_node_labels or n.label+'_'+str(i) in goal_node_labels] + [('connected', 'l1', 'l'+str(i)) for i, n in enumerate(graph.nodes) if n.label in goal_node_labels or n.label+'_'+str(i) in goal_node_labels]


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
def grasp_heuristics(ifco_pose, object_pose, bounding_box, uncertainty_offset):
    #ifco dimensions
    xd = 0.37/2 
    yd = 0.57/2 
    #boundary width from which to go for a wall_grasp
    e = 0.13

    # corner1_in_base = object_pose.dot(tra.translation_matrix([bounding_box.x/2 + uncertainty_offset, bounding_box.y/2 + uncertainty_offset, 0]))
    # corner2_in_base = object_pose.dot(tra.translation_matrix([bounding_box.x/2 + uncertainty_offset, -bounding_box.y/2 - uncertainty_offset, 0]))
    # corner3_in_base = object_pose.dot(tra.translation_matrix([-bounding_box.x/2 - uncertainty_offset, -bounding_box.y/2 - uncertainty_offset, 0]))
    # corner4_in_base = object_pose.dot(tra.translation_matrix([-bounding_box.x/2 - uncertainty_offset, bounding_box.y/2 + uncertainty_offset, 0]))
    corner1_in_base = object_pose.dot(tra.translation_matrix([bounding_box.x/2, bounding_box.y/2, 0]))
    corner2_in_base = object_pose.dot(tra.translation_matrix([bounding_box.x/2, -bounding_box.y/2, 0]))
    corner3_in_base = object_pose.dot(tra.translation_matrix([-bounding_box.x/2, -bounding_box.y/2, 0]))
    corner4_in_base = object_pose.dot(tra.translation_matrix([-bounding_box.x/2, bounding_box.y/2, 0]))


    max_x = max([corner1_in_base[0,3], corner2_in_base[0,3], corner3_in_base[0,3], corner4_in_base[0,3]])
    min_x = min([corner1_in_base[0,3], corner2_in_base[0,3], corner3_in_base[0,3], corner4_in_base[0,3]])
    max_y = max([corner1_in_base[1,3], corner2_in_base[1,3], corner3_in_base[1,3], corner4_in_base[1,3]])
    min_y = min([corner1_in_base[1,3], corner2_in_base[1,3], corner3_in_base[1,3], corner4_in_base[1,3]])

    # ifco_x = ifco_pose[0,3]
    # ifco_y = ifco_pose[1,3]

    # if abs(max_x - ifco_x) > abs(min_x - ifco_x):
        # x = max_x - ifco_x
    # else:
        # x = min_x - ifco_x

    # if abs(max_y - ifco_y) > abs(min_y - ifco_y):
        # y = max_y - ifco_y
    # else:
        # y = min_y - ifco_y
    object_pos_in_ifco = tra.translation_from_matrix((object_pose - ifco_pose))
    x = object_pos_in_ifco[0]
    y = object_pos_in_ifco[1]
    
    elongated_x = (max_x - min_x)/(max_y - min_y) > 2
    elongated_y = (max_y - min_y)/(max_x - min_x) > 2

    #                      ROBOT
    #                      wall4         
    #                 =============
    #          wall3  |           |  wall1
    #                 |           |
    #                 =============
    #                      wall2         
    #
    print("GRASP HEURISTICS x:" + str(x) + " y:" + str(y))
    if abs(x) < xd - e and abs(y) < yd - e:
        return "SurfaceGrasp", "NoWall"
    elif y > yd - e:
        if x > xd - e and not elongated_x:
            return "WallGrasp", "wall2"
        else:
            return "WallGrasp", "wall1"
    elif y < -yd + e:
        if x < -xd + e and not elongated_x:
            return "WallGrasp", "wall4" 
        else:
            return "WallGrasp", "wall3" 
    elif x > xd - e:
        if y < -yd + e and not elongated_y:
            return "WallGrasp", "wall3" 
        else:
            return "WallGrasp", "wall2" 
    elif x < -xd + e:
        if y > yd - e and not elongated_y:
            return "WallGrasp", "wall1" 
        else:
            return "WallGrasp", "wall4" 
    else:
        return "object not in ifco", "NoWall"

# ================================================================================================
def get_wall_tf(ifco_tf, wall_id):
    # rotate the tf following the wall id see figure in grasp_heuristics()
    wall4_tf = ifco_tf.dot(tra.rotation_matrix(
                    math.radians(90), [1, 0, 0]))
    rotation_angle = 0
    if wall_id == 'wall1':
        rotation_angle = -90
    elif wall_id == 'wall2':
        rotation_angle = 180
    elif wall_id == 'wall3':
        rotation_angle = 90
    elif wall_id == 'wall4':
        return wall4_tf
    return tra.concatenate_matrices(wall4_tf, tra.rotation_matrix(
                    math.radians(rotation_angle), [0, 1, 0]))

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
if __name__ == '__main__':

    print(sys.argv)
    parser = argparse.ArgumentParser(description='Turn path in graph into hybrid automaton.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ros_service_call', action='store_true', default = False,
                        help='Whether to send the hybrid automaton to a ROS service called /update_hybrid_automaton.')
    parser.add_argument('--file_output', action='store_true', default = False,
                        help='Whether to write the hybrid automaton to a file called hybrid_automaton.xml.')
    #grasp_choices = ["any", "EdgeGrasp", "WallGrasp", "SurfaceGrasp"]
    # parser.add_argument('--grasp', choices=grasp_choices, default=grasp_choices[0],
    #                     help='Which grasp type to use.')
    # parser.add_argument('--grasp_id', type=int, default=-1,
    #                    help='Which specific grasp to use. Ignores any values < 0.')
    parser.add_argument('--rviz', action='store_true', default = False,
                        help='Whether to send marker messages that can be seen in RViz and represent the chosen grasping motion.')
    parser.add_argument('--robot_base_frame', type=str, default = 'base_link',
                        help='Name of the robot base frame.')
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

    while not rospy.is_shutdown():
        marker_pub.publish(markers_rviz)
        for i, f in enumerate(frames_rviz):
            br.sendTransform(tra.translation_from_matrix(f),
                             tra.quaternion_from_matrix(f),
                             rospy.Time.now(),
                             "dbg_frame_" + str(i),
                             robot_base_frame)

        r.sleep()

# ================================================================================================
# ================================================================================================
