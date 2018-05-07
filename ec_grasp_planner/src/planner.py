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

import tf
from tf import transformations as tra
import numpy as np

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from subprocess import call
from hybrid_automaton_msgs import srv as ha_srv

from std_msgs.msg import Header

from geometry_graph_msgs.msg import Graph

from ec_grasp_planner import srv as plan_srv

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

from pregrasp_msgs import srv as vision_srv

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
        self.grasp_type = req.grasp_type
        grasp_choices = ["any", "WallGrasp", "SurfaceGrasp", "EdgeGrasp"]
        if self.grasp_type not in grasp_choices:
            raise rospy.ServiceException("grasp_type not supported. Choose from [any,WallGrasp,SurfaceGrasp,EdgeGrasp]")
            return

        #todo: more failure handling here for bad service parameters

        self.handarm_params = handarm_parameters.__dict__[req.handarm_type]()

        rospy.wait_for_service('compute_ec_graph')

        try:
            call_vision = rospy.ServiceProxy('compute_ec_graph', vision_srv.ComputeECGraph)
            res = call_vision(self.object_type)
            graph = res.graph
            objects = res.objects.objects
        except rospy.ServiceException, e:
            raise rospy.ServiceException("Vision service call failed: %s" % e)
            return plan_srv.RunGraspPlannerResponse("")

        robot_base_frame = self.args.robot_base_frame
        object_frame = objects[0].transform

        time = rospy.Time(0)
        graph.header.stamp = time
        object_frame.header.stamp = time
        bounding_box = objects[0].boundingbox

        if self.args.frame_convention == 'new':
            frame_convention = 0
        elif self.args.frame_convention == 'old':
            frame_convention = 1
        else:
            raise rospy.ServiceException("Frame convention not supported, use new or old!")
            return plan_srv.RunGraspPlannerResponse("")

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

            print("Received graph with {} nodes and {} edges.".format(len(graph.nodes), len(graph.edges)))

            # Find a path in the ECE graph
            hand_node_id = [n.label for n in graph.nodes].index("Positioning")
            object_node_id = [n.label for n in graph.nodes].index("Slide")

            grasp_path = find_a_path(hand_node_id, object_node_id, graph, self.grasp_type, verbose=True)

            rospy.sleep(0.3)

        # --------------------------------------------------------
        # Turn grasp into hybrid automaton
        ha, self.rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, graph_in_base, object_in_base,
                                                                self.handarm_params, self.object_type, frame_convention)

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
def create_surface_grasp(object_frame, support_surface_frame, handarm_params, object_type):

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
    hand_synergy = params['hand_closing_synergy']
    down_speed = params['down_speed']
    up_speed = params['up_speed']
    go_down_velocity = params['go_down_velocity']

    # Set the initial pose above the object

    goal_ = np.copy(object_frame)

    #we rotate the frame to align the hand with the long object axis and the table surface
    x_axis = object_frame[:3,0]
    z_axis = support_surface_frame[:3,2]
    y_axis = np.cross(z_axis,x_axis)

    goal_[:3,0] = x_axis
    goal_[:3,1] = y_axis/np.linalg.norm(y_axis)
    goal_[:3,2] = z_axis

    goal_ =  goal_.dot(hand_transform)

    #the grasp frame is symmetrical - we flip so that the hand axis always points away from the robot base
    zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    if goal_[0][0]<0:
        goal_ = goal_.dot(zflip_transform)

    # hand pose above object
    pre_grasp_pose = goal_.dot(pregrasp_transform)

    # Set the directions to use task space controller with
    dirDown = tra.translation_matrix([0, 0, -down_speed])
    dirUp = tra.translation_matrix([0, 0, up_speed])

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
        ha.InterpolatedHTransformControlMode(dirDown,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="world",
                                             v_max=go_down_velocity))

    force  = np.array([0, 0, downward_force, 0, 0, 0])
    # 3b. Switch when goal is reached
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'softhand_close',
                                                 goal = force,
                                                 norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion = "THRESH_UPPER_BOUND",
                                                 goal_is_relative = '1',
                                                 frame_id = 'world',
                                                 port = '2'))

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
        control_sequence.append(ha.SimpleRBOHandControlMode(name  = 'softhand_close', goal = np.array([1])))


    # 4b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration = hand_closing_time))

    # 5. Rotate hand after closing and before lifting it up
    # relative to current hand pose
    control_sequence.append(
        ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate', goal_is_relative='1', ))

    # 5b. Switch when hand rotated
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp', controller='PostGraspRotate', epsilon='0.01', goal_is_relative='1', reference_frame = 'EE'))

    # 6. Lift upwards
    control_sequence.append(ha.InterpolatedHTransformControlMode(dirUp, controller_name = 'GoUpHTransform', name = 'GoUp', goal_is_relative='1', reference_frame="world"))
 
    # 6b. Switch when joint is reached
    control_sequence.append(ha.FramePoseSwitch('GoUp', 'GoDropOff', controller = 'GoUpHTransform', epsilon = '0.01', goal_is_relative='1', reference_frame="world"))
     
    # 7. Go to dropOFF
    control_sequence.append(ha.JointControlMode(drop_off_config, controller_name = 'GoToDropJointConfig', name = 'GoDropOff'))
 
    # 7.b  Switch when joint is reached
    control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'finished', controller = 'GoToDropJointConfig', epsilon = str(math.radians(7.))))    
 
    # 8. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name  = 'finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller( name = 'JointSpaceController', type = 'JointController', goal  = np.zeros(7),
                                   goal_is_relative = 1, v_max = '[0,0]', a_max = '[0,0]'))
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
    slide_velocity = params['slide_velocity']
    hand_closing_duration = params['hand_closing_duration']

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
    dirDown = tra.translation_matrix([0, 0, -down_dist])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirDown,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="world",
                                             v_max=go_down_velocity))

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
    dirLift = tra.translation_matrix([0, 0, lift_dist])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirLift, controller_name='Lift1', goal_is_relative='1', name="LiftHand",
                                             reference_frame="world"))

    # 3b. We switch after a short time as this allows us to do a small, precise lift motion
    # TODO partners: this can be replaced by a frame pose switch if your robot is able to do small motions precisely
    control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=0.2))

    # 4. Go towards the wall to slide object to wall
    dirWall = tra.translation_matrix([0, 0, -sliding_dist])
    #TODO sliding_distance should be computed from wall and hand frame.

    # slide direction is given by the normal of the wall
    dirWall[:3, 3] = wall_frame[:3, :3].dot(dirWall[:3, 3])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirWall, controller_name='SlideToWall', goal_is_relative='1',
                                             name="SlideToWall", reference_frame="world",
                                             v_max=slide_velocity))

    # 4b. Switch when the f/t sensor is triggered with normal force from wall
    # TODO arne: needs tuning
    force = np.array([0, 0, wall_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', 'softhand_close', 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame, port='2'))

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
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.SimpleRBOHandControlMode(name  = 'softhand_close', goal = np.array([1])))


    # 5b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration=hand_closing_duration))

    # 6. Move hand after closing and before lifting it up
    # relative to current hand pose
    control_sequence.append(
        ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate',
                                 goal_is_relative='1', ))

    # 6b. Switch when hand reaches post grasp pose
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp', controller='PostGraspRotate', epsilon='0.01',
                                               goal_is_relative='1', reference_frame='EE'))

    # 7. Lift upwards (+z in world frame)
    dirUp = tra.translation_matrix([0, 0, up_dist])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirUp, controller_name='GoUpHTransform', name='GoUp', goal_is_relative='1',
                                             reference_frame="world"))

    # 7b. Switch when lifting motion is completed
    control_sequence.append(
        ha.FramePoseSwitch('GoUp', 'GoDropOff', controller='GoUpHTransform', epsilon='0.01', goal_is_relative='1',
                           reference_frame="world"))

    # 8. Go to drop off configuration
    control_sequence.append(
        ha.JointControlMode(drop_off_config, controller_name='GoToDropJointConfig', name='GoDropOff'))

    # 8.b  Switch when configuration is reached
    control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'finished', controller='GoToDropJointConfig',
                                                        epsilon=str(math.radians(7.))))

    # 9. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name='finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller(name='JointSpaceController', type='JointController', goal=np.zeros(7),
                                  goal_is_relative=1, v_max='[0,0]', a_max='[0,0]'))
    finishedMode.set(finishedSet)
    control_sequence.append(finishedMode)

    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames

# ================================================================================================
def create_edge_grasp(object_frame, support_surface_frame, edge_frame, handarm_params, object_type):

    # Get the parameters from the handarm_parameters.py file
    if (object_type in handarm_params['wall_grasp']):
        params = handarm_params['edge_grasp'][object_type]
    else:
        params = handarm_params['edge_grasp']['object']

    initial_jointConf = params['initial_goal']

    # transformation between hand and EC frame (which is positioned like object and oriented like edge?) at grasp time
    hand_transform = params['hand_transform']

    # transformation to apply after grasping
    post_grasp_transform = params['post_grasp_transform']


    downward_force = params['table_force']
    up_dist = params['up_dist']
    down_dist = params['down_dist']
    pre_approach_transform = params['pre_approach_transform']
    drop_off_config = params['drop_off_config']
    hand_over_config = params['hand_over_config']
    hand_over_force = params['hand_over_force']

    go_down_velocity = params['go_down_velocity']
    slide_velocity = params['slide_velocity']
    hand_closing_duration = params['hand_closing_duration']

    # Get the pose above the object
    global rviz_frames
    rviz_frames = []

    # this is the EC frame. It is positioned like object and oriented to the edge?
    # TODO rename EC_FRAME to initial_slide_frame
    ec_frame = np.copy(edge_frame)
    ec_frame[:3, 3] = tra.translation_from_matrix(object_frame)
    # apply hand transformation
    ec_frame = (ec_frame.dot(hand_transform))

    #the grasp frame is symmetrical - we flip so that the hand axis always points away from the robot base
    # TODO check for edge EC if this stands
    zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    if ec_frame[0][0]<0:
        goal_ = ec_frame.dot(zflip_transform)


    # the pre-approach pose should be:
    # - floating above the object
    # -- position above the object depend the relative location of the edge to the object and the object bounding box
    #   TODO define pre_approach_transform(egdeTF, objectTF, objectBB)
    # - fingers pointing toward the edge (edge x-axis orthogonal to palm frame x-axis)
    # - palm normal perpendicualr to surface normal
    pre_approach_pose = ec_frame.dot(pre_approach_transform)


    # Rviz debug frames
    rviz_frames.append(edge_frame)
    #rviz_frames.append(ec_frame)
    #rviz_frames.append(position_behind_object)
    rviz_frames.append(ec_frame)
    rviz_frames.append(pre_approach_pose)


    # use the default synergy
    hand_synergy = 1

    control_sequence = []

    # 0. initial position above table
    control_sequence.append(
        ha.JointControlMode(initial_jointConf, name='InitialJointConfig', controller_name='initialJointCtrl'))

    # 0b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('InitialJointConfig', 'PreGrasp', controller='initialJointCtrl',
                                                        epsilon=str(math.radians(7.))))

    # 1. Go above the object
    # TODO hand pose relative to object should depend on bounding box size and edge location
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(pre_approach_pose, controller_name='GoAboveObject', goal_is_relative='0',
                                             name="PreGrasp"))

    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'GoDown', controller='GoAboveObject', epsilon='0.01'))

    # 2. Go down onto the object/table, in world frame
    dirDown = tra.translation_matrix([0, 0, -down_dist])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirDown,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="world",
                                             v_max=go_down_velocity))

    # 2b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'Pause_1',
                                                 goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))

    # gravity compensation before sliding due to safety reasons
    # safety_force_3 triggers at the begining of slide controller on WAM
    # bypasseing issue by pause between Control Modes
    # TODO: fix issue on WAM controller
    control_sequence.append(ha.GravityCompensationMode(name = 'Pause_1'))
    # time switch - to end pause
    control_sequence.append(ha.TimeSwitch('Pause_1', 'SlideToEdge', duration=0.02))

    # 4. Go towards the edge to slide object to edge
    #this is the tr from ec_frame to edge frame in ec_frame
    delta = np.linalg.inv(edge_frame).dot(ec_frame)

    #this is the distance to the edge, given by the z axis of the edge frame
    dist = delta[2,3]

    # handPalm pose on the edge right before grasping
    hand_on_edge_pose = ec_frame.dot(tra.translation_matrix([dist, 0, 0]))

    # direction toward the edge and distance without any rotation in worldFrame
    dirEdge = np.identity(4)
    # relative distance from initial slide position to final hand on the edge position
    distance_to_edge = hand_on_edge_pose - ec_frame
    # TODO might need tuning based on object and palm frame
    dirEdge[:3, 3] = tra.translation_from_matrix(distance_to_edge)
    # no lifting motion applied while sliding
    dirEdge[2, 3] = 0

    # slide direction is given by the x-axis of the edge
    rviz_frames.append(hand_on_edge_pose)
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirEdge, controller_name='SlideToEdge', goal_is_relative='1',
                                             name="SlideToEdge", reference_frame="world",
                                             v_max=slide_velocity))

    # 4b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('SlideToEdge', 'softhand_close',
                                               controller='SlideToEdge',
                                               epsilon='0.01',
                                               reference_frame="world",
                                               goal_is_relative='1'))

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
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.SimpleRBOHandControlMode(name='softhand_close', goal=np.array([1])))

    # 5b time switch for closing hand  to post grasp rotation to increase graps success - TODO might be not needed check it
    control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration=hand_closing_duration))

    # 6. Move hand after closing and before lifting it up
    # relative to current hand pose
    control_sequence.append(
        ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate',
                                 goal_is_relative='1', ))

    # 6b. Switch when hand reaches post grasp pose
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp', controller='PostGraspRotate', epsilon='0.01',
                                               goal_is_relative='1', reference_frame='EE'))

    # 7. Lift upwards (+z in world frame)
    dirUp = tra.translation_matrix([0, 0, up_dist])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirUp, controller_name='GoUpHTransform', name='GoUp', goal_is_relative='1',
                                             reference_frame="world"))

    # 7b. Switch when lifting motion is completed
    control_sequence.append(
        ha.FramePoseSwitch('GoUp', 'GoDropOff', controller='GoUpHTransform', epsilon='0.01', goal_is_relative='1',
                           reference_frame="world"))

    # 8. Go to hand over configuration configuration
    control_sequence.append(
        ha.JointControlMode(hand_over_config, controller_name='GoToDropJointConfig', name='GoDropOff'))

    # 8.b  Switch when configuration is reached
    control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'wait_for_handover', controller='GoToDropJointConfig',
                                                        epsilon=str(math.radians(7.))))

    # 9. Block joints to hold object in air util human takes over
    waitForHandoverMode = ha.ControlMode(name='wait_for_handover')
    waitForHandoverSet = ha.ControlSet()
    waitForHandoverSet.add(ha.Controller(name='JointSpaceController', type='JointController', goal=np.zeros(7),
                                  goal_is_relative=1, v_max='[0,0]', a_max='[0,0]'))
    waitForHandoverMode.set(waitForHandoverSet)
    control_sequence.append(waitForHandoverMode)

    # wait until forece is exerted by human on EE while taking the object to release object
    control_sequence.append(ha.ForceTorqueSwitch('wait_for_handover', 'softhand_open', name='human_interaction_handover', goal=np.zeros(6),
                                                   norm_weights=np.array([1, 1, 1, 0, 0, 0]), jump_criterion='0',
                                                   epsilon=hand_over_force, negate='1'))

    # open hand and go into gravity compensation mode
    control_sequence.append(ha.ControlMode(name='softhand_open'))

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
def hybrid_automaton_from_motion_sequence(motion_sequence, graph, T_robot_base_frame, T_object_in_base, handarm_params, object_type, frame_convention):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))

    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label
    #grasp_frame = grasp_frames[grasp_type]

    print("Creating hybrid automaton for object {} and grasp type {}.".format(object_type, grasp_type))
    if grasp_type == 'EdgeGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        edge_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        edge_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(edge_frame_node.transform))

        #we need to flip the z axis in the old convention
        if frame_convention == 1:

            x_flip_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(180.0), [1, 0, 0]))
            support_surface_frame = support_surface_frame.dot(x_flip_transform)

            x_flip_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(-90.0), [1, 0, 0]))
            edge_frame = edge_frame.dot(x_flip_transform)


        return create_edge_grasp(T_object_in_base, support_surface_frame, edge_frame, handarm_params, object_type)
    elif grasp_type == 'WallGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        wall_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        wall_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(wall_frame_node.transform))

        #we need to flip the z axis in the old convention
        if frame_convention == 1:
            x_flip_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(180.0), [1, 0, 0]))
            support_surface_frame = support_surface_frame.dot(x_flip_transform)
            x_flip_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(-90.0), [1, 0, 0]))
            wall_frame = wall_frame.dot(x_flip_transform)


        return create_wall_grasp(T_object_in_base, support_surface_frame, wall_frame, handarm_params, object_type)
    elif grasp_type == 'SurfaceGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))

        #we need to flip the z axis in the old convention
        if frame_convention == 1:
            x_flip_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(180.0), [1, 0, 0]))
            support_surface_frame = support_surface_frame.dot(x_flip_transform)

        return create_surface_grasp(T_object_in_base, support_surface_frame, handarm_params, object_type)

    else:
        raise "Unknown grasp type: ", grasp_type

# ================================================================================================
def find_a_path(hand_start_node_id, object_start_node_id, graph, goal_node_labels, verbose = False):
    locations = ['l'+str(i) for i in range(len(graph.nodes))]

    connections = [('connected', 'l'+str(e.node_id_start), 'l'+str(e.node_id_end)) for e in graph.edges]

    # grasping_locations_tmp = [('is_grasping_location', 'l'+str(i)) for i, n in enumerate(graph.nodes) if n.label in goal_node_labels or n.label+'_'+str(i) in goal_node_labels]
    # random edge selection
    # tuning purpose we might use l5 ec
    grasping_locations_tmp = ['l'+str(i) for i, n in enumerate(graph.nodes) if n.label in goal_node_labels or n.label+'_'+str(i) in goal_node_labels]
    print("graps locations: {}".format(grasping_locations_tmp))
    grasping_locations = [('is_grasping_location', np.random.choice(grasping_locations_tmp))]
    print("random graps location: {}".format(grasping_locations))

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
    parser.add_argument('--frame_convention', type=str, default = 'new',
                        help='Whether to use new or old frame convention of EC graph. Use old to run ec_graphs where z-axis is pointing down')
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
