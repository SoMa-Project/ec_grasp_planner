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
from numpy.random.mtrand import choice
from tf import transformations as tra
import numpy as np

from grasp_success_estimator import RESPONSES

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

from enum import Enum

import pyddl

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')
sys.path.append(pkg_path + '/../hybrid-automaton-tools-py/')
import hatools.components as ha
import hatools.cookbook as cookbook
import tf_conversions.posemath as pm

import handarm_parameters

import multi_object_params as mop

markers_rviz = MarkerArray()
frames_rviz = []


class FailureCases(Enum):
    MASS_ESTIMATION_NO_OBJECT = 3
    MASS_ESTIMATION_TOO_MANY = 4


class Reaction:
    def __init__(self, object_type, strategy, object_params):
        self.label_to_cm = {
            'REEXECUTE': 'failure_rerun_',
            'REPLAN': 'failure_replan_',
            'CONTINUE': None,  # depends on the non-failure case
        }

        # contains all failure cases that depend on the mass parameter
        self.mass_depended = [FailureCases.MASS_ESTIMATION_NO_OBJECT, FailureCases.MASS_ESTIMATION_TOO_MANY]

        self.reactions = {}
        if 'reactions' in object_params[object_type][strategy]:
            reaction_param = object_params[object_type][strategy]['reactions']
            for rp in reaction_param:
                self.reactions[FailureCases[rp.upper()]] = reaction_param[rp]

        self.object_type = object_type
        self.strategy = strategy
        if 'mass' in object_params[object_type]:
            self.mass = object_params[object_type]['mass']
        else:
            self.mass = None

    # returns true iff no reaction was defined at all
    def is_no_reaction(self):
        return not self.reactions  # reactions is not empty

    def cm_name_for(self, failure_case, default):

        if self.is_no_reaction():
            # No reactions defined => there is no special (failure) control mode => return the default CM
            return default

        if failure_case not in self.reactions:
            raise KeyError("No reaction parameter set for {} ({}, {})".format(failure_case, self.object_type,
                                                                              self.strategy))

        if self.mass is None and failure_case in self.mass_depended:
            raise ValueError("No mass parameter set for {} ({}, {})".format(failure_case, self.object_type,
                                                                            self.strategy))

        if self.label_to_cm[self.reactions[failure_case]] is None:
            # There is no special (failure) control mode required (e.g. CONTINUE) return the default
            return default

        # return the name of the cm for the respective failure case
        return self.label_to_cm[self.reactions[failure_case]] + str(failure_case.value)


class GraspPlanner:

    # maximum amount of time (in seconds) that the success estimator has for its measurements per control mode.
    success_estimator_timeout = 10.0

    def __init__(self, args):
        # initialize the ros node
        rospy.init_node('ec_planner')
        s = rospy.Service('run_grasp_planner', plan_srv.RunGraspPlanner, lambda msg: self.handle_run_grasp_planner(msg))
        self.tf_listener = tf.TransformListener()
        self.args = args
        # initialize the object-EC selection handler class
        self.multi_object_handler = mop.multi_object_params(args.object_params_file)

        # clean initialization of planner service arguments in __init__
        self.object_type = ""
        self.grasp_type = ""
        self.handarm_params = None


    # ------------------------------------------------------------------------------------------------
    def handle_run_grasp_planner(self, req):

        print('Handling grasp planner service call')
        self.object_type = req.object_type
        self.grasp_type = req.grasp_type

        # Check for bad service parameters (we don't have to check for object since we always have a default 'object')
        grasp_choices = ["Any", "WallGrasp", "SurfaceGrasp", "EdgeGrasp"]
        if self.grasp_type not in grasp_choices:
            raise rospy.ServiceException("grasp_type {0} not supported. Choose from {1}".format(self.grasp_type,
                                                                                                grasp_choices))

        heuristic_choices = ["Deterministic", "Probabilistic", "Random"]
        if req.object_heuristic_function not in heuristic_choices:
            raise rospy.ServiceException("heuristic {0} not supported. Choose from {1}".format(req.object_heuristi,
                                                                                               heuristic_choices))

        if req.handarm_type not in handarm_parameters.__dict__:
            raise rospy.ServiceException("handarm type not supported. Did you add {0} to handarm_parameters.py".format(
                req.handarm_type))

        # load hand arm parameters set in handarm_parameters.py
        self.handarm_params = handarm_parameters.__dict__[req.handarm_type]()
        # check if the handarm parameters aren't containing any contradicting information or bugs because of non-copying
        self.handarm_params.checkValidity()

        try:
            print('Wait for vision service')
            rospy.wait_for_service('compute_ec_graph', timeout=30)
        except rospy.ROSException as e:
            raise rospy.ServiceException("Vision service call unavailable: %s" % e)

        try:
            print('Call vision service now...!')
            call_vision = rospy.ServiceProxy('compute_ec_graph', vision_srv.ComputeECGraph)
            res = call_vision(self.object_type)
            graph = res.graph
            objects = res.objects.objects
        except rospy.ServiceException as e:
            raise rospy.ServiceException("Vision service call failed: %s" % e)

        if not objects:
            print("Vision: No object was detected")
            return plan_srv.RunGraspPlannerResponse(success=False,
                                                    hybrid_automaton_xml="Vision: No object was detected",
                                                    chosen_object_idx=-1)

        robot_base_frame = self.args.robot_base_frame
        object_frame = objects[0].transform

        time = rospy.Time(0)
        graph.header.stamp = time
        object_frame.header.stamp = time # TODO why do we need to change the time in the header?
        bounding_box = objects[0].boundingbox

        # build list of objects
        object_list = []
        for o in objects:
            obj_tmp = {}
            obj_tmp['type'] = self.object_type

            # the TF must be in the same reference frame as the EC frames
            # Get the object frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, o.transform.header.frame_id, time,
                                              rospy.Duration(2.0))
            camera_in_base = self.tf_listener.asMatrix(robot_base_frame, o.transform.header)
            object_in_camera = pm.toMatrix(pm.fromMsg(o.transform.pose))
            object_in_base = camera_in_base.dot(object_in_camera)
            obj_tmp['frame'] = object_in_base
            obj_tmp['bounding_box'] = o.boundingbox
            object_list.append(obj_tmp)

        # selecting list of goal nodes based on requested strategy type
        if self.grasp_type == "Any":
            goal_node_labels = ['SurfaceGrasp', 'WallGrasp', 'EdgeGrasp']
        else:
            goal_node_labels = [self.grasp_type]

        # print(" *** goal node lables: {} ".format(goal_node_labels))

        node_list = [n for i, n in enumerate(graph.nodes) if n.label in goal_node_labels]

        # Get the geometry graph frame in robot base frame
        self.tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, time, rospy.Duration(2.0))
        graph_in_base_transform = self.tf_listener.asMatrix(robot_base_frame, graph.header)

        self.tf_listener.waitForTransform('ifco', robot_base_frame, time, rospy.Duration(2.0))
        (ifco_in_base_translation, ifco_in_base_rot) = self.tf_listener.lookupTransform('ifco', robot_base_frame, rospy.Time(0)) #transform_msg_to_homogenous_tf()
        tf_transformer = tf.TransformerROS()
        ifco_in_base_transform = tf_transformer.fromTranslationRotation(ifco_in_base_translation, ifco_in_base_rot)
        print ifco_in_base_transform

        # we assume that all objects are on the same plane, so all EC can be exploited for any of the objects
        (chosen_object_idx, chosen_node_idx) = self.multi_object_handler.process_objects_ecs(object_list,
                                                                                     node_list,
                                                                                     graph_in_base_transform,
                                                                                     ifco_in_base_transform,
                                                                                     req.object_heuristic_function,
                                                                                     self.handarm_params
                                                                                     )

        if chosen_object_idx < 0:
            return plan_srv.RunGraspPlannerResponse(success=False,
                                                    hybrid_automaton_xml="No feasible trajectory was found",
                                                    chosen_object_idx=-1)

        chosen_object = object_list[chosen_object_idx]
        chosen_node = node_list[chosen_node_idx]
        # print(" * object type: {}, ec type: {}, heuristc funciton type: {}".format(chosen_object['type'], chosen_node.label, req.object_heuristic_function))


        # --------------------------------------------------------
        # Get grasp from graph representation
        grasp_path = None
        while grasp_path is None:
            # Get the geometry graph frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, time, rospy.Duration(2.0))
            graph_in_base = self.tf_listener.asMatrix(robot_base_frame, graph.header)

            # Get the object frame in robot base frame
            object_in_base = chosen_object['frame']

            # Find a path in the ECE graph
            hand_node_id = [n.label for n in graph.nodes].index("Positioning")
            object_node_id = [n.label for n in graph.nodes].index("Slide")

            #strategy selection based on grasp type
            # grasp_path = find_a_path(hand_node_id, object_node_id, graph, self.grasp_type, verbose=True)

            #strategy selection based on object-ec-hand heuristic
            grasp_path = find_a_path(hand_node_id, object_node_id, graph, [chosen_node], verbose=True)

            rospy.sleep(0.3)

        # --------------------------------------------------------
        # Turn grasp into hybrid automaton
        ha, self.rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, graph_in_base, object_in_base,
                                                                     self.handarm_params, self.object_type,
                                                                     self.multi_object_handler.get_object_params(),
                                                                     self.multi_object_handler.get_alternative_behavior(
                                                                         chosen_object_idx, chosen_node_idx))

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

        ha_as_xml = ha.xml()
        return plan_srv.RunGraspPlannerResponse(success=ha_as_xml != "",
                                                hybrid_automaton_xml=ha_as_xml,
                                                chosen_object_idx=chosen_object_idx if ha_as_xml != "" else -1,
                                                chosen_node=chosen_node)


# ================================================================================================
def create_surface_grasp(object_frame, support_surface_frame, handarm_params, object_type, object_params,
                         alternative_behavior):

    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['surface_grasp']:
        params = handarm_params['surface_grasp'][object_type]
    else:
        params = handarm_params['surface_grasp']['object']

    reaction = Reaction(object_type, 'SurfaceGrasp', object_params)

    hand_transform = params['hand_transform']
    pregrasp_transform = params['pregrasp_transform']
    post_grasp_transform = params['post_grasp_transform']

    drop_off_config = params['drop_off_config']
    downward_force = params['downward_force']
    hand_closing_time = params['hand_closing_duration']
    hand_synergy = params['hand_closing_synergy']
    down_dist = params['down_dist']
    up_dist = params['up_dist']
    go_down_velocity = params['go_down_velocity']
    pre_grasp_velocity = params['pre_grasp_velocity']

    # Set the initial pose above the object
    goal_ = np.copy(object_frame) #TODO: this should be support_surface_frame
    goal_[:3,3] = tra.translation_from_matrix(object_frame)
    goal_ =  goal_.dot(hand_transform)

    # the grasp frame is symmetrical - check which side is nicer to reach
    # this is a hacky first version for our WAM
    zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    if goal_[0][0] < 0:
        goal_ = goal_.dot(zflip_transform)

    # hand pose above object
    pre_grasp_pose = goal_.dot(pregrasp_transform)

    # Set the directions to use TRIK controller with
    dirDown = tra.translation_matrix([0, 0, -down_dist])
    # half the distance we want to achieve since we do two consecutive lifts
    dirUp_2 = tra.translation_matrix([0, 0, up_dist/2.0])

    # force threshold that if reached will trigger the closing of the hand
    force = np.array([0, 0, downward_force, 0, 0, 0])
    # the 1 in softhand_close_1 represents a surface grasp. This way the strategy is encoded in the HA.
    mode_name_hand_closing = 'softhand_close_1_0'

    # Set the frames to visualize with RViz
    rviz_frames = [object_frame, goal_, pre_grasp_pose]

    # assemble controller sequence
    control_sequence = []

    # # 1. Go above the object - PreGrasp
    # control_sequence.append(
    #     ha.InterpolatedHTransformControlMode(pre_grasp_pose, controller_name='GoAboveIFCO', goal_is_relative='0',
    #                                          name='Pre_preGrasp'))
    #
    # # 1b. Switch when hand reaches the goal pose
    # control_sequence.append(ha.FramePoseSwitch('Pre_preGrasp', 'PreGrasp', controller='GoAboveIFCO', epsilon='0.01'))

    # TODO get rid of this workaround since we actually have joint space planning capabilities
    initial_jointConf = params['initial_goal']

    # 0. initial position above ifco
    control_sequence.append(
        ha.JointControlMode(initial_jointConf, name='InitialJointConfig', controller_name='initialJointCtrl'))

    # 0b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('InitialJointConfig', 'softhand_preshape_1_1',
                                                        controller='initialJointCtrl', epsilon=str(math.radians(7.))))

    # #################################

    # 0. trigger pre-shaping the hand (if there is a synergy). The first 1 in the name represents a surface grasp.
    control_sequence.append(ha.BlockJointControlMode(name='softhand_preshape_1_1'))
    control_sequence.append(ha.TimeSwitch('softhand_preshape_1_1', 'PreGrasp',
                                          duration=0.2))  # time to trigger pre-shape

    # 1. Go above the object - PreGrasp
    if alternative_behavior is not None and 'pre_grasp' in alternative_behavior:

        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        pre_grasp_velocity = params['pre_grasp_joint_velocity']
        goal_traj = alternative_behavior['pre_grasp'].get_trajectory()

        print(goal_traj)  # TODO Check if the dimensions are correct and the via points are as expected
        control_sequence.append(ha.JointControlMode(goal_traj, name='PreGrasp', controller_name='GoAboveObject',
                                                    goal_is_relative='0',
                                                    v_max=pre_grasp_velocity))

            #ha.InterpolatedJointController(goal_traj, name='PreGrasp', v_max=pre_grasp_velocity))
    else:
        # we can use the original motion
        control_sequence.append(ha.InterpolatedHTransformControlMode(pre_grasp_pose,
                                                                     name='PreGrasp',
                                                                     controller_name='GoAboveObject',
                                                                     goal_is_relative='0',
                                                                     v_max=pre_grasp_velocity))

    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'ReferenceMassMeasurement', controller='GoAboveObject',
                                               epsilon='0.01'))

    # TODO add a time switch and short waiting time before the reference measurement is actually done?
    # 2. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))  # TODO use gravity comp instead?

    # 2b. Switches when reference measurement was done
    # 2b.1 Successful reference measurement
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.REFERENCE_MEASUREMENT_SUCCESS.value]),
                                              ))

    # 2b.2 The grasp success estimator module is inactive
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 2b.3 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('ReferenceMassMeasurement', 'GoDown',
                                          duration=GraspPlanner.success_estimator_timeout))

    # 2b.4 There is no special switch for unknown error response (estimator signals REFERENCE_MEASUREMENT_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.

    # 3. Go down onto the object - Godown
    if alternative_behavior is not None and 'go_down' in alternative_behavior:
        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        # Go down onto the object (joint controller + relative world frame motion)

        go_down_velocity = params['pre_grasp_joint_velocity']
        goal_traj = alternative_behavior['go_down'].get_trajectory()

        print("GOAL_TRAJ", goal_traj)  # TODO Check if the dimensions are correct and the via points are as expected
        control_sequence.append(ha.JointControlMode(goal_traj, name='GoDown', controller_name='GoDown',
                                                    goal_is_relative='0',
                                                    v_max=go_down_velocity))

        # Relative motion that ensures that the actual force/torque threshold is reached
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(tra.translation_matrix([0, 0, -10]),
                                                 controller_name='GoDownFurther',
                                                 goal_is_relative='1',
                                                 name="GoDownFurther",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

        # Force/Torque switch for the additional relative go down
        control_sequence.append(ha.ForceTorqueSwitch('GoDownFurther',
                                                     mode_name_hand_closing,
                                                     goal=force,
                                                     norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion="THRESH_UPPER_BOUND",
                                                     goal_is_relative='1',
                                                     frame_id='world',
                                                     port='2'))

    else:
        # Go down onto the object (relative in world frame)
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(dirDown,
                                                 controller_name='GoDown',
                                                 goal_is_relative='1',
                                                 name="GoDown",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

    # 3b. Switch when goal is reached
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 mode_name_hand_closing,
                                                 goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))

    # 4. Maintain the position
    desired_displacement = np.array([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]])

    force_gradient = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.005],
                               [0.0, 0.0, 0.0, 1.0]])

    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    if handarm_params['isForceControllerAvailable']:
        control_sequence.append(ha.HandControlMode_ForceHT(name=mode_name_hand_closing, synergy=hand_synergy,
                                                           desired_displacement=desired_displacement,
                                                           force_gradient=force_gradient,
                                                           desired_force_dimension=desired_force_dimension))
    else:
        # if hand is not RBO then create general hand closing mode?
        control_sequence.append(ha.SimpleRBOHandControlMode(goal=np.array([1.0]), name=mode_name_hand_closing))

    # 4b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'PostGraspRotate', duration=hand_closing_time))

    # 5. Rotate hand after closing and before lifting it up relative to current hand pose
    control_sequence.append(ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate',
                                                     name='PostGraspRotate', goal_is_relative='1'))

    # 5b. Switch when hand rotated
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp_1', controller='PostGraspRotate',
                                               epsilon='0.01', goal_is_relative='1', reference_frame='EE'))

    # 6. Lift upwards a little bit (half way up)
    control_sequence.append(ha.InterpolatedHTransformControlMode(dirUp_2, controller_name='GoUpHTransform',
                                                                 name='GoUp_1', goal_is_relative='1',
                                                                 reference_frame="world"))

    # 6b. Switch when joint configuration (half way up) is reached
    control_sequence.append(ha.FramePoseSwitch('GoUp_1', 'EstimationMassMeasurement', controller='GoUpHTransform',
                                               epsilon='0.01', goal_is_relative='1', reference_frame="world"))

    # 7. Measure the mass again and estimate number of grasped objects (grasp success estimation)
    control_sequence.append(ha.BlockJointControlMode(name='EstimationMassMeasurement'))

    # 7b. Switches after estimation measurement was done
    target_cm_okay = 'GoUp_2'

    # 7b.1 No object was grasped => go to failure mode.
    target_cm_estimation_no_object = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_NO_OBJECT, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_no_object,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_NO_OBJECT.value]),
                                              ))

    # 7b.2 More than one object was grasped => failure
    target_cm_estimation_too_many = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_TOO_MANY, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_too_many,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_TOO_MANY.value]),
                                              ))

    # 7b.3 Exactly one object was grasped => success (continue lifting the object and go to drop off)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_OKAY.value]),
                                              ))

    # 7b.4 The grasp success estimator module is inactive => directly continue lifting the object and go to drop off
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 7b.5 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('EstimationMassMeasurement', target_cm_okay,
                                          duration=GraspPlanner.success_estimator_timeout))

    # 7b.6 There is no special switch for unknown error response (estimator signals ESTIMATION_RESULT_UNKNOWN_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.

    # 8. After estimation measurement control modes.
    extra_failure_cms = set()
    if target_cm_estimation_no_object != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_no_object)
    if target_cm_estimation_too_many != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_too_many)

    for cm in extra_failure_cms:
        if cm.startswith('failure_rerun'):
            # 8.1 Failure control mode representing grasping failure, which might be corrected by re-running the plan.
            control_sequence.append(ha.GravityCompensationMode(name=cm))
        if cm.startswith('failure_replan'):
            # 8.2 Failure control mode representing grasping failure, which can't be corrected and requires to re-plan.
            control_sequence.append(ha.GravityCompensationMode(name=cm))

    # 8.3 Success control mode. Lift hand even further
    control_sequence.append(ha.InterpolatedHTransformControlMode(dirUp_2, controller_name='GoUpHTransform',
                                                                 name=target_cm_okay, goal_is_relative='1',
                                                                 reference_frame="world"))

    # 8.3b Switch when joint configuration is reached
    control_sequence.append(ha.FramePoseSwitch(target_cm_okay, 'GoDropOff', controller='GoUpHTransform', epsilon='0.01',
                                               goal_is_relative='1', reference_frame="world"))

    # 9. Go to dropOFF location
    control_sequence.append(ha.JointControlMode(drop_off_config, controller_name='GoToDropJointConfig',
                                                name='GoDropOff'))

    # 9.b  Switch when joint is reached
    control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'finished', controller = 'GoToDropJointConfig',
                                                        epsilon=str(math.radians(7.))))

    # 10. Block joints to finish motion and hold object in air
    control_sequence.append(ha.BlockJointControlMode(name='finished'))

    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames


# ================================================================================================
def create_wall_grasp(object_frame, support_surface_frame, wall_frame, handarm_params, object_type, object_params,
                      alternative_behavior):

    # Get the parameters from the handarm_parameters.py file
    if (object_type in handarm_params['wall_grasp']):
        params = handarm_params['wall_grasp'][object_type]
    else:
        params = handarm_params['wall_grasp']['object']

    reaction = Reaction(object_type, 'WallGrasp', object_params)

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
    rviz_frames.append(pre_approach_pose)
    rviz_frames.append(ec_frame)
    rviz_frames.append(wall_frame)

    # use the default synergy
    hand_synergy = 1

    control_sequence = []

    # 0. initial position above ifco
    control_sequence.append(
        ha.JointControlMode(initial_jointConf, name='InitialJointConfig', controller_name='initialJointCtrl'))

    # 0b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('InitialJointConfig', 'softhand_preshape_2_1',
                                                        controller='initialJointCtrl', epsilon=str(math.radians(7.))))

    # 0.5 trigger pre-shaping the hand (if there is a synergy). The 2 in the name represents a wall grasp.
    control_sequence.append(ha.BlockJointControlMode(name='softhand_preshape_2_1'))
    control_sequence.append(ha.TimeSwitch('softhand_preshape_2_1', 'PreGrasp', duration=0.5))  # time for pre-shape

    # 1. Go above the object
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(pre_approach_pose, controller_name='GoAboveObject', goal_is_relative='0',
                                             name="PreGrasp"))

    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'ReferenceMassMeasurement', controller='GoAboveObject',
                                               epsilon='0.01'))

    # TODO add a time switch and short waiting time before the reference measurement is actually done?
    # 2. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))  # TODO use gravity comp instead?

    # 2b. Switches when reference measurement was done
    # 2b.1 Successful reference measurement
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.REFERENCE_MEASUREMENT_SUCCESS.value]),
                                              ))

    # 2b.2 The grasp success estimator module is inactive
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 2b.3 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('ReferenceMassMeasurement', 'GoDown',
                                          duration=GraspPlanner.success_estimator_timeout))

    # 2b.4 There is no special switch for unknown error response (estimator signals REFERENCE_MEASUREMENT_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.

    # 3. Go down onto the object/table, in world frame
    dirDown = tra.translation_matrix([0, 0, -down_dist])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirDown,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="world",
                                             v_max=go_down_velocity))

    # 3b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'LiftHand',
                                                 goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))

    # TODO: remove 4 and 4b if hand should slide on surface
    # OR duration=0.0
    # 4. Lift upwards so the hand doesn't slide on table surface
    dirLift = tra.translation_matrix([0, 0, lift_dist])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirLift, controller_name='Lift1', goal_is_relative='1', name="LiftHand",
                                             reference_frame="world"))

    # 4b. We switch after a short time as this allows us to do a small, precise lift motion
    # TODO partners: this can be replaced by a frame pose switch if your robot is able to do small motions precisely
    control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=0.2))

    # 5. Go towards the wall to slide object to wall
    dirWall = tra.translation_matrix([0, 0, -sliding_dist])
    #TODO sliding_distance should be computed from wall and hand frame.

    # slide direction is given by the normal of the wall
    dirWall[:3, 3] = wall_frame[:3, :3].dot(dirWall[:3, 3])
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(dirWall, controller_name='SlideToWall', goal_is_relative='1',
                                             name="SlideToWall", reference_frame="world",
                                             v_max=slide_velocity))

    # 5b. Switch when the f/t sensor is triggered with normal force from wall
    # TODO arne: needs tuning
    force = np.array([0, 0, wall_force, 0, 0, 0])
    # The 2 in softhand_close_2 represents a wall grasp. This way the strategy is encoded in the HA.
    # The 0 encodes the synergy id
    mode_name_hand_closing = 'softhand_close_2_0'

    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', mode_name_hand_closing, 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame, port='2'))

    # 6. Maintain contact while closing the hand
    if handarm_params['isForceControllerAvailable']:
        # apply force on object while closing the hand
        # TODO arne: validate these values
        desired_displacement = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        force_gradient = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
        desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        control_sequence.append(ha.HandControlMode_ForceHT(name=mode_name_hand_closing, synergy=hand_synergy,
                                                           desired_displacement=desired_displacement,
                                                           force_gradient=force_gradient,
                                                           desired_force_dimension=desired_force_dimension))
    else:
        # just close the hand
        control_sequence.append(ha.close_rbohand())

    # 6b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'PostGraspRotate', duration=hand_closing_duration))

    # 7. Move hand after closing and before lifting it up
    # relative to current hand pose
    control_sequence.append(
        ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate',
                                 goal_is_relative='1', ))

    # 7b. Switch when hand reaches post grasp pose
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp_1', controller='PostGraspRotate',
                                               epsilon='0.01', goal_is_relative='1', reference_frame='EE'))

    # 8. Lift upwards (+z in world frame)
    # split the distance we want to achieve since we do two consecutive lifts
    scale_up = 0.7
    dir_up1 = tra.translation_matrix([0, 0, scale_up * up_dist])
    dir_up2 = tra.translation_matrix([0, 0, (1.0 - scale_up) * up_dist])

    control_sequence.append(ha.InterpolatedHTransformControlMode(dir_up1, controller_name='GoUpHTransform',
                                                                 name='GoUp_1', goal_is_relative='1',
                                                                 reference_frame="world"))

    # 8b. Switch when joint configuration (half way up) is reached
    control_sequence.append(ha.FramePoseSwitch('GoUp_1', 'EstimationMassMeasurement', controller='GoUpHTransform',
                                               epsilon='0.01', goal_is_relative='1', reference_frame="world"))

    # 9. Measure the mass again and estimate number of grasped objects (grasp success estimation)
    control_sequence.append(ha.BlockJointControlMode(name='EstimationMassMeasurement'))

    # 9b. Switches after estimation measurement was done
    target_cm_okay = 'GoUp_2'

    # 9b.1 No object was grasped => go to failure mode.
    target_cm_estimation_no_object = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_NO_OBJECT,
                                                          default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_no_object,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_NO_OBJECT.value]),
                                              ))

    # 9b.2 More than one object was grasped => failure
    target_cm_estimation_too_many = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_TOO_MANY, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_too_many,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_TOO_MANY.value]),
                                              ))

    # 9b.3 Exactly one object was grasped => success (continue lifting the object and go to drop off)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_OKAY.value]),
                                              ))

    # 9b.4 The grasp success estimator module is inactive => directly continue lifting the object and go to drop off
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 9b.5 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('EstimationMassMeasurement', target_cm_okay,
                                          duration=GraspPlanner.success_estimator_timeout))

    # 9b.6 There is no special switch for unknown error response (estimator signals ESTIMATION_RESULT_UNKNOWN_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.

    # 10. After estimation measurement control modes.
    extra_failure_cms = set()
    if target_cm_estimation_no_object != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_no_object)
    if target_cm_estimation_too_many != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_too_many)

    for cm in extra_failure_cms:
        if cm.startswith('failure_rerun'):
            # 10.1 Failure control mode representing grasping failure, which might be corrected by re-running the plan.
            control_sequence.append(ha.GravityCompensationMode(name=cm))
        if cm.startswith('failure_replan'):
            # 10.2 Failure control mode representing grasping failure, which can't be corrected and requires to re-plan.
            control_sequence.append(ha.GravityCompensationMode(name=cm))

    # 10.3 Success control mode. Lift hand even further
    control_sequence.append(ha.InterpolatedHTransformControlMode(dir_up2, controller_name='GoUpHTransform',
                                                                 name=target_cm_okay, goal_is_relative='1',
                                                                 reference_frame="world"))

    # 10.3b Switch when joint configuration is reached
    control_sequence.append(ha.FramePoseSwitch(target_cm_okay, 'GoDropOff', controller='GoUpHTransform', epsilon='0.01',
                                               goal_is_relative='1', reference_frame="world"))

    # 11. Go to drop off configuration
    control_sequence.append(
        ha.JointControlMode(drop_off_config, controller_name='GoToDropJointConfig', name='GoDropOff'))

    # 11.b Switch when configuration is reached
    control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'finished', controller='GoToDropJointConfig',
                                                        epsilon=str(math.radians(7.))))

    # 12. Block joints to finish motion and hold object in air
    control_sequence.append(ha.BlockJointControlMode(name='finished'))

    return cookbook.sequence_of_modes_and_switches_with_safety_features(control_sequence), rviz_frames


# ================================================================================================
def transform_msg_to_homogeneous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]),
                  tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))


# ================================================================================================
def homogeneous_tf_to_pose_msg(htf):
    return Pose(position=Point(*tra.translation_from_matrix(htf).tolist()),
                orientation=Quaternion(*tra.quaternion_from_matrix(htf).tolist()))


# ================================================================================================
def get_node_from_actions(actions, action_name, graph):
    return graph.nodes[[int(m.sig[1][1:]) for m in actions if m.name == action_name][0]]


# ================================================================================================
def hybrid_automaton_from_motion_sequence(motion_sequence, graph, T_robot_base_frame, T_object_in_base, handarm_params,
                                          object_type, object_params, alternative_behavior=None):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))

    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label
    #grasp_frame = grasp_frames[grasp_type]

    print("Creating hybrid automaton for object {} and grasp type {}.".format(object_type, grasp_type))
    if grasp_type == 'EdgeGrasp':
        raise ValueError("Edge grasp is not supported yet")
        #support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        #support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        #edge_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        #edge_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(edge_frame_node.transform))
        return create_edge_grasp(T_object_in_base, support_surface_frame, edge_frame, handarm_params, object_type,
                                 object_params, alternative_behavior)
    elif grasp_type == 'WallGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogeneous_tf(support_surface_frame_node.transform))
        wall_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        wall_frame = T_robot_base_frame.dot(transform_msg_to_homogeneous_tf(wall_frame_node.transform))
        return create_wall_grasp(T_object_in_base, support_surface_frame, wall_frame, handarm_params, object_type,
                                 object_params, alternative_behavior)
    elif grasp_type == 'SurfaceGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogeneous_tf(support_surface_frame_node.transform))
        return create_surface_grasp(T_object_in_base, support_surface_frame, handarm_params, object_type, object_params,
                                    alternative_behavior)
    else:
        raise ValueError("Unknown grasp type: {}".format(grasp_type))


# ================================================================================================
def find_a_path(hand_start_node_id, object_start_node_id, graph, goal_node_list, verbose = False):
    locations = ['l'+str(i) for i in range(len(graph.nodes))]

    connections = [('connected', 'l'+str(e.node_id_start), 'l'+str(e.node_id_end)) for e in graph.edges]

    # strategy selectio based on grasp type without heuristic
    # grasping_locations = [('is_grasping_location', 'l'+str(i)) for i, n in enumerate(graph.nodes) if n.label in goal_node_labels or n.label+'_'+str(i) in goal_node_labels]

    # strategy selection based on heuristics
    grasping_locations = [('is_grasping_location', 'l'+str(i)) for i, n in enumerate(graph.nodes) if n in goal_node_list or n.label+'_'+str(i) in goal_node_list]

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
        msg.pose = homogeneous_tf_to_pose_msg(f)

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
        msg.points.append(homogeneous_tf_to_pose_msg(f1).position)
        msg.points.append(homogeneous_tf_to_pose_msg(f2).position)

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
    parser.add_argument('--object_params_file', type=str, default='object_param.yaml',
                        help='Name of the file containing object specific parameters (e.g. for object-EC selection when multiple objects are present)')

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

    r = rospy.Rate(5)

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
