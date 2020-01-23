#!/usr/bin/env python
import rospy
import numpy as np
import sys
import argparse

import tf
from tf import transformations as tra

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

from hybrid_automaton_msgs import srv as ha_srv

from std_msgs.msg import Header

from ec_grasp_planner import srv as plan_srv

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

from pregrasp_msgs import srv as vision_srv

from enum import Enum

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')
sys.path.append(pkg_path + '/../hybrid-automaton-tools-py/')
import hatools.components as ha
import hatools.cookbook as cookbook
import tf_conversions.posemath as pm

import handarm_parameters
import math

import PISAHandRecipes
import RBOHandRecipesKUKA
import RBOHandRecipesWAM
import PISAGripperRecipes
import ClashHandRecipes
import TransportRecipesWAM
import TransportRecipesKUKA
import PlacementRecipes
import multi_object_params as mop
from GraspFrameRecipes import get_derived_corner_grasp_frames

from planner_utils import *

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
        angleOfAttack_req = req.angle_of_attack
        wristAngle_req = req.wrist_angle

        # Check for bad service parameters (we don't have to check for object since we always have a default 'object')
        grasp_choices = ["Any", "WallGrasp", "SurfaceGrasp", "EdgeGrasp", "CornerGrasp"]
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
            object_list_msg = res.objects
        except rospy.ServiceException as e:
            raise rospy.ServiceException("Vision service call failed: %s" % e)

        if not object_list_msg.objects:
            print("Vision: No object was detected")
            return plan_srv.RunGraspPlannerResponse(success=False,
                                                    hybrid_automaton_xml="Vision: No object was detected",
                                                    chosen_object_idx=-1)

        robot_base_frame = self.args.robot_base_frame       

        time = rospy.Time(0)
        graph.header.stamp = time
        
        self.tf_listener.waitForTransform(robot_base_frame, "/ifco", time, rospy.Duration(2.0))
        ifco_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, time, "ifco"))
        print(ifco_in_base)

        # Naming of camera frame is a convention established by the vision nodes so no reason to pretend otherwise
        self.tf_listener.waitForTransform(robot_base_frame, "/camera", time, rospy.Duration(2.0))
        camera_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, time, "camera")) 

        # selecting list of goal nodes based on requested strategy type
        if self.grasp_type == "Any":
            goal_node_labels = ['SurfaceGrasp', 'WallGrasp', 'EdgeGrasp']
            robot_name = rospy.get_param('/planner_gui/robot')
            if robot_name == 'WAM':  # TODO remove if CornerGrasp is integrated to KUKA as well
                goal_node_labels.append('CornerGrasp')           
        else:
            goal_node_labels = [self.grasp_type]

        # print(" *** goal node lables: {} ".format(goal_node_labels))

        node_list = [n for i, n in enumerate(graph.nodes) if n.label in goal_node_labels]

        # Get the geometry graph frame in robot base frame
        self.tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, time, rospy.Duration(2.0))
        graph_in_base = self.tf_listener.asMatrix(robot_base_frame, graph.header)

        # get pre grasp transforms in object frame for both grasp types 
        pre_grasps_in_object_frame = get_pre_grasp_transforms(self.handarm_params, self.object_type)
        SG_pre_grasp_in_object_frame = pre_grasps_in_object_frame[0]
        WG_pre_grasp_in_object_frame = pre_grasps_in_object_frame[1]
        CG_pre_grasp_in_object_frame = pre_grasps_in_object_frame[2]

        # we assume that all objects are on the same plane, so all EC can be exploited for any of the objects
        (chosen_object, chosen_node_idx, pre_grasp_pose_in_base) = self.multi_object_handler.process_objects_ecs(
                                                                                    self.object_type,
                                                                                    node_list,
                                                                                    graph_in_base,
                                                                                    ifco_in_base,
                                                                                    SG_pre_grasp_in_object_frame,
                                                                                    WG_pre_grasp_in_object_frame,
                                                                                    CG_pre_grasp_in_object_frame,
                                                                                    req.object_heuristic_function,
                                                                                    self.grasp_type,                                                                                    
                                                                                    object_list_msg,
                                                                                    self.handarm_params,
                                                                                    angleOfAttack_req,
                                                                                    wristAngle_req )

        if pre_grasp_pose_in_base is None:
            # No grasp found
            return plan_srv.RunGraspPlannerResponse(success=False,
                                                    hybrid_automaton_xml="No feasible trajectory was found",
                                                    chosen_object_idx=-1)

        # lookup chosen node
        chosen_node = node_list[chosen_node_idx]

        # --------------------------------------------------------
        # Turn grasp into hybrid automaton
        ha, self.rviz_frames = hybrid_automaton_from_object_EC_combo(chosen_node,
                                                                     chosen_object,
                                                                     pre_grasp_pose_in_base,
                                                                     graph_in_base,
                                                                     self.handarm_params,
                                                                     req.handarm_type,
                                                                     self.multi_object_handler.get_object_params(),
                                                                     self.multi_object_handler.get_alternative_behavior(
                                                                         chosen_object['index'], chosen_node_idx)
                                                                     )

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
                                                chosen_object_idx=chosen_object['index'] if ha_as_xml != "" else -1,
                                                chosen_node=chosen_node)


# ================================================================================================
def get_hand_recipes(handarm_type, robot_name):
    if "PISAHand" in handarm_type:
        return PISAHandRecipes
    elif "PISAGripper" in handarm_type:
        return PISAGripperRecipes
    elif "RBOHand" in handarm_type:
        if robot_name == "WAM":
            return RBOHandRecipesWAM
        elif robot_name == "KUKA":
            return RBOHandRecipesKUKA
        else:
            raise Exception("Unknown robot_name: " + robot_name)
    elif "ClashHand" in handarm_type:
        return ClashHandRecipes
    else:
        raise Exception("Unknown handarm_type: " + handarm_type)


# ================================================================================================
def get_handarm_params(handarm_params, object_type, grasp_type):
    # Get the parameters from the handarm_parameters.py file

    if object_type in handarm_params[grasp_type]:
        return handarm_params[grasp_type][object_type]

    return handarm_params[grasp_type]['object']


# ================================================================================================
def get_pre_grasp_transforms(handarm_params, object_type):
    # Returns the initial pre_grasp transforms for wall, surface and corner grasp depending on the object type and the
    # hand in the object frame
    
    # Surface grasp pre_grasp transform SG_pre_grasp_transform
    params = get_handarm_params(handarm_params, object_type, "SurfaceGrasp")

    hand_transform = params['hand_transform']
    ee_in_goal_frame = params['ee_in_goal_frame']
    pre_approach_transform = params['pre_approach_transform']

    SG_pre_grasp_transform = (hand_transform.dot(pre_approach_transform)).dot(ee_in_goal_frame)

    # Wall grasp pre_grasp transform WG_pre_grasp_transform
    params = get_handarm_params(handarm_params, object_type, "WallGrasp")

    hand_transform = params['hand_transform']
    pre_approach_transform = params['pre_approach_transform']

    WG_pre_grasp_transform = hand_transform.dot(pre_approach_transform)

    # Corner grasp pre_grasp transform CG_pre_grasp_transform
    params = get_handarm_params(handarm_params, object_type, "CornerGrasp")
    pre_approach_transform = params['pre_approach_transform']
    hand_transform = params['hand_transform']

    CG_pre_grasp_transform = hand_transform.dot(pre_approach_transform)

    return SG_pre_grasp_transform, WG_pre_grasp_transform, CG_pre_grasp_transform


# ================================================================================================
def transform_msg_to_homogeneous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]),
                  tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))


# ================================================================================================
def homogeneous_tf_to_pose_msg(htf):
    return Pose(position=Point(*tra.translation_from_matrix(htf).tolist()),
                orientation=Quaternion(*tra.quaternion_from_matrix(htf).tolist()))


# ================================================================================================
def hybrid_automaton_from_object_EC_combo(chosen_node, chosen_object, pre_grasp_pose, graph_in_base, handarm_params,
                                          handarm_type, object_params, alternative_behavior=None):

    print("Creating hybrid automaton for object {} and grasp type {}.".format(chosen_object['type'], chosen_node.label))


    # Set the frames to visualize with RViz
    rviz_frames = [chosen_object['frame'], pre_grasp_pose]
    grasp_type = chosen_node.label
    robot_name = rospy.get_param('/planner_gui/robot')

    if grasp_type == 'EdgeGrasp':
        raise ValueError("Edge grasp is not supported yet")
    elif grasp_type == 'WallGrasp':  
        wall_frame = graph_in_base.dot(transform_msg_to_homogeneous_tf(chosen_node.transform))
        grasping_recipe = get_hand_recipes(handarm_type, robot_name).create_wall_grasp(chosen_object, wall_frame,
                                                                                       handarm_params, pre_grasp_pose,
                                                                                       alternative_behavior)
        rviz_frames.append(wall_frame)       
    elif grasp_type == 'SurfaceGrasp':
        grasping_recipe = get_hand_recipes(handarm_type, robot_name).create_surface_grasp(chosen_object, handarm_params,
                                                                                          pre_grasp_pose,
                                                                                          alternative_behavior)
    elif grasp_type == 'CornerGrasp':
        corner_frame = graph_in_base.dot(transform_msg_to_homogeneous_tf(chosen_node.transform))
        corner_frame_alpha_zero = get_derived_corner_grasp_frames(corner_frame, chosen_object['frame'])[1]
        grasping_recipe = get_hand_recipes(handarm_type, robot_name).create_corner_grasp(chosen_object,
                                                                                         corner_frame_alpha_zero,
                                                                                         handarm_params,
                                                                                         pre_grasp_pose,
                                                                                         alternative_behavior)
        rviz_frames.append(corner_frame)
        rviz_frames.append(corner_frame_alpha_zero)

        # rviz_frames.append(pre_approach_transform)
        # rviz_frames.append(ec_frame)
        # rviz_frames.append(corner_frame)
        # rviz_frames.append(corner_frame_alpha_zero)

    else:
        raise ValueError("Unknown grasp type: {}".format(grasp_type))

    if robot_name == 'WAM':
        # TODO removed transportation for grasp funnel evaluation
        transport_recipe = TransportRecipesWAM.get_transport_recipe(chosen_object, handarm_params, Reaction(chosen_object['type'], grasp_type, object_params), FailureCases, grasp_type)
        # return cookbook.sequence_of_modes_and_switches_with_safety_features(grasping_recipe + transport_recipe), rviz_frames
        return cookbook.sequence_of_modes_and_switches_with_safety_features(grasping_recipe), rviz_frames
    elif robot_name == 'KUKA':
        transport_recipe = TransportRecipesKUKA.get_transport_recipe(chosen_object, handarm_params, Reaction(chosen_object['type'], grasp_type, object_params), FailureCases, grasp_type)
        placement_recipe = PlacementRecipes.get_placement_recipe(chosen_object, handarm_params, grasp_type)
        return cookbook.sequence_of_modes_and_switches(grasping_recipe + transport_recipe + placement_recipe), rviz_frames
    else:
        raise ValueError("No robot named {}".format(robot_name))


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
