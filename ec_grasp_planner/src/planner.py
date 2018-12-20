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

import smach
import smach_ros

import tf
from numpy.random.mtrand import choice
from tf import transformations as tra
import numpy as np
from numpy.linalg import inv

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

from target_selection_in_ifco import srv as target_selection_srv
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

import PISAHandRecipes
import RBOHandRecipes
import PISAGripperRecipes
import ClashHandRecipes
import TransportRecipes
import multi_object_params as mop

markers_rviz = MarkerArray()
frames_rviz = []

use_ocado_heuristic = True

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
            objectList = res.objects # TODO: unify this with the object list used below
        except rospy.ServiceException as e:
            raise rospy.ServiceException("Vision service call failed: %s" % e)

        if not objects:
            print("No object was detected")
            return plan_srv.RunGraspPlannerResponse("", -1)

        robot_base_frame = self.args.robot_base_frame       

        time = rospy.Time(0)
        graph.header.stamp = time
        
        # this will only be set if the Ocado heuristic + kinematic feasibility check is used
        pre_grasp_pose_in_base = None  

        self.tf_listener.waitForTransform(robot_base_frame, "/ifco", time, rospy.Duration(2.0))
        ifco_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, time, "ifco"))
        print ifco_in_base

        # Naming of camera frame is a convention established by the vision nodes so no reason to pretend otherwise
        self.tf_listener.waitForTransform(robot_base_frame, "/camera", time, rospy.Duration(2.0))
        camera_in_base = self.tf_listener.asMatrix(robot_base_frame, Header(0, time, "camera")) 

        # selecting list of goal nodes based on requested strategy type
        if self.grasp_type == "Any":
            goal_node_labels = ['SurfaceGrasp', 'WallGrasp', 'EdgeGrasp']
        else:
            goal_node_labels = [self.grasp_type]

        # print(" *** goal node lables: {} ".format(goal_node_labels))

        node_list = [n for i, n in enumerate(graph.nodes) if n.label in goal_node_labels]

        if use_ocado_heuristic:
            # call target_selection_node

            camera_in_ifco = inv(ifco_in_base).dot(camera_in_base)
            camera_in_ifco_msg = pm.toMsg(pm.fromMatrix(camera_in_ifco))

            # get pre grasp transforms in object frame for both grasp types 
            SG_pre_grasp_transform, WG_pre_grasp_transform = get_pre_grasp_transforms(self.handarm_params, self.object_type)

            SG_success_rate, WG_success_rate = get_success_rate(self.handarm_params, self.object_type)

            SG_pre_grasp_in_object_frame_msg = pm.toMsg(pm.fromMatrix(SG_pre_grasp_transform))
            WG_pre_grasp_in_object_frame_msg = pm.toMsg(pm.fromMatrix(WG_pre_grasp_transform))
            ifco_in_base_msg = pm.toMsg(pm.fromMatrix(ifco_in_base))

            graspable_with_any_hand_orientation = self.handarm_params[self.object_type]['graspable_with_any_hand_orientation']

            call_heuristic = rospy.ServiceProxy('target_selection', target_selection_srv.TargetSelection)

            # TODO: Make the heuristic respect the ["Any", "WallGrasp", "SurfaceGrasp", "EdgeGrasp"] selection
            res = call_heuristic(objectList, camera_in_ifco_msg, SG_pre_grasp_in_object_frame_msg, WG_pre_grasp_in_object_frame_msg, ifco_in_base_msg, graspable_with_any_hand_orientation, SG_success_rate, WG_success_rate)

            if res.grasp_type == 'no_grasp':
                print "GRASP HEURISTICS: NO SUITABLE TARGET FOUND!!"
                print "EXECUTING SURFACE GRASP OR WALL GRASP ON WALL 1 (IF ONLY WALL GRASP WAS SELECTED) ON RANDOM OBJECT"
                #TODO: implement a better way for the planner to handle this case?
                chosen_node_idx = 0
            else:
                pre_grasp_pose_in_base = pm.toMatrix(pm.fromMsg(res.pre_grasp_pose_in_base_frame))
                object_in_base = pm.toMatrix(pm.fromMsg(res.target_pose_in_base_frame))

                if res.grasp_type == 's':
                    chosen_node_idx = 0 
                else:
                    chosen_node_idx = int(res.grasp_type[1]) - (self.grasp_type == 'WallGrasp') # the node list will only contain wall grasps if the grasp type is a wall grasp

                print "GRASP HEURISTICS " + self.grasp_type
                if res.grasp_type[0] == 'w':
                    print "CHOSEN WALL IS WALL " + res.grasp_type[1]            
            
            chosen_object_idx = res.chosen_object_idx 

            chosen_object = {}
            chosen_object['type'] = self.object_type

            # the TF must be in the same reference frame as the EC frames
            # Get the object frame in robot base frame
            object_in_camera = pm.toMatrix(pm.fromMsg(objects[chosen_object_idx].transform.pose))
            object_in_base = camera_in_base.dot(object_in_camera)
            chosen_object['frame'] = object_in_base
            chosen_object['bounding_box'] = objects[chosen_object_idx].boundingbox
        else:
            # use TUB code
            # build list of objects
            object_list = []
            for o in objects:
                obj_tmp = {}
                obj_tmp['type'] = self.object_type

                # the TF must be in the same reference frame as the EC frames
                # Get the object frame in robot base frame
                object_in_camera = pm.toMatrix(pm.fromMsg(o.transform.pose))
                object_in_base = camera_in_base.dot(object_in_camera)
                obj_tmp['frame'] = object_in_base
                obj_tmp['bounding_box'] = o.boundingbox
                object_list.append(obj_tmp)

            # Get the geometry graph frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, time, rospy.Duration(2.0))
            graph_in_base = self.tf_listener.asMatrix(robot_base_frame, graph.header)
            

            # we assume that all objects are on the same plane, so all EC can be exploited for any of the objects
            (chosen_object_idx, chosen_node_idx) = self.multi_object_handler.process_objects_ecs(object_list,
                                                                                        node_list,
                                                                                        graph_in_base,
                                                                                        ifco_in_base,
                                                                                        req.object_heuristic_function
                                                                                        )
            chosen_object = object_list[chosen_object_idx]
        
        chosen_node = node_list[chosen_node_idx]

        # --------------------------------------------------------
        # Get grasp from graph representation
        grasp_path = None
        while grasp_path is None:
            # Get the geometry graph frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, time, rospy.Duration(2.0))
            graph_in_base = self.tf_listener.asMatrix(robot_base_frame, graph.header)

            # Get the object frame in robot base frame
            if not use_ocado_heuristic:
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

        
        wall_frame_node = get_node_from_actions(grasp_path, 'grasp_object', graph)
        wall_frame = graph_in_base.dot(transform_msg_to_homogeneous_tf(wall_frame_node.transform))


        ha, self.rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, graph_in_base, object_in_base, chosen_object['bounding_box'],
                                                                self.handarm_params, self.object_type, wall_frame, req.handarm_type, self.multi_object_handler.get_object_params(), pre_grasp_pose_in_base)
                                                
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
        return plan_srv.RunGraspPlannerResponse(ha_as_xml, chosen_object_idx if ha_as_xml != "" else -1, chosen_node)


# ================================================================================================
def transform_msg_to_homogeneous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]),
                  tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))

# ================================================================================================
def getParam(obj_type_params, obj_params, paramKey):
    param = obj_type_params.get(paramKey)
    if param is None:
        param = obj_params.get(paramKey)
    if param is None:
         raise Exception("Param: " + paramKey + " does not exist for this object and there is no generic value defined")
    return param

# ================================================================================================
def get_hand_recipes(handarm_type):
    if handarm_type == "PISAHandKUKA":
        return PISAHandRecipes
    elif handarm_type == "PISAGripperKUKA":
        return PISAGripperRecipes
    elif handarm_type == "RBOHandO2KUKA":
        return RBOHandRecipes
    elif handarm_type == "ClashHandKUKA":
        return ClashHandRecipes
    else:
        raise Exception("Unknown handarm_type: " + handarm_type)

# ================================================================================================
def get_pre_grasp_transforms(handarm_params, object_type):
    #returns the initial pre_grasp transforms for wall grasp and surface grasp depending on the object type and the hand
    
    #surface grasp pre_grasp transform SG_pre_grasp_transform
    obj_type_params = {}
    obj_params = {}
    if (object_type in handarm_params['surface_grasp']):            
        obj_type_params = handarm_params['surface_grasp'][object_type]
    if 'object' in handarm_params['surface_grasp']:
        obj_params = handarm_params['surface_grasp']['object']

    hand_transform = getParam(obj_type_params, obj_params, 'hand_transform')    
    ee_in_goal_frame = getParam(obj_type_params, obj_params, 'ee_in_goal_frame')

    SG_pre_grasp_transform = hand_transform.dot(ee_in_goal_frame)

    #wall grasp pre_grasp transform WG_pre_grasp_transform
    obj_type_params = {}
    obj_params = {}
    if (object_type in handarm_params['wall_grasp']):            
        obj_type_params = handarm_params['wall_grasp'][object_type]
    if 'object' in handarm_params['wall_grasp']:
        obj_params = handarm_params['wall_grasp']['object']

    hand_transform = getParam(obj_type_params, obj_params, 'hand_transform')    
    pre_approach_transform = getParam(obj_type_params, obj_params, 'pre_approach_transform')

    WG_pre_grasp_transform = hand_transform.dot(pre_approach_transform)

    return SG_pre_grasp_transform, WG_pre_grasp_transform

# ================================================================================================
def get_success_rate(handarm_params, object_type):
    #returns the success rate for the surface and wall grasp for the specific object type
    
    #surface grasp success rate
    obj_type_params = {}
    obj_params = {}
    if (object_type in handarm_params['surface_grasp']):            
        obj_type_params = handarm_params['surface_grasp'][object_type]
    if 'object' in handarm_params['surface_grasp']:
        obj_params = handarm_params['surface_grasp']['object']

    SG_success_rate = getParam(obj_type_params, obj_params, 'success_rate')        

    #wall grasp success rate
    obj_type_params = {}
    obj_params = {}
    if (object_type in handarm_params['wall_grasp']):            
        obj_type_params = handarm_params['wall_grasp'][object_type]
    if 'object' in handarm_params['wall_grasp']:
        obj_params = handarm_params['wall_grasp']['object']

    WG_success_rate = getParam(obj_type_params, obj_params, 'success_rate')        

    return SG_success_rate, WG_success_rate

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
def hybrid_automaton_from_motion_sequence(motion_sequence, graph, T_robot_base_frame, T_object_in_base, bounding_box, handarm_params, object_type, wall_frame, handarm_type, object_params, pre_grasp_pose_in_base = None):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))

    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label

    print("Creating hybrid automaton for object {} and grasp type {}.".format(object_type, grasp_type))
    if grasp_type == 'EdgeGrasp':
        raise ValueError("Edge grasp is not supported yet")
    elif grasp_type == 'WallGrasp':        
        grasping_recipe, rviz_frames = get_hand_recipes(handarm_type).create_wall_grasp(T_object_in_base, bounding_box, wall_frame, handarm_params, object_type, object_params, pre_grasp_pose_in_base)        
    elif grasp_type == 'SurfaceGrasp':
        grasping_recipe, rviz_frames = get_hand_recipes(handarm_type).create_surface_grasp(T_object_in_base, bounding_box, handarm_params, object_type, object_params, pre_grasp_pose_in_base)
    else:
        raise ValueError("Unknown grasp type: {}".format(grasp_type))

    transport_recipe = TransportRecipes.get_transport_recipe(handarm_params)

    robot_name = rospy.get_param('/planner_gui/robot')
    if robot_name == 'WAM':
        return cookbook.sequence_of_modes_and_switches_with_safety_features(grasping_recipe + transport_recipe), rviz_frames
    elif robot_name == 'KUKA':
        return cookbook.sequence_of_modes_and_switches(grasping_recipe + transport_recipe), rviz_frames
    else:
        raise ValueError("No robot named {}".format(robot_name))

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
def get_wall_tf(ifco_tf, wall_id):
    # rotate the tf following the wall id
    #     
    #                      ROBOT
    #                      wall4         
    #                 ================
    #                 |     ^ y       |
    #                 |     |         |
    #          wall3  |     |--->x    |  wall1
    #                 |               |
    #                 ================
    #                      wall2         
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
