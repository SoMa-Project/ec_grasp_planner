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
from numpy.linalg import inv

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
            objectList = res.objects
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

            # Get the object frame in robot base frame
            self.tf_listener.waitForTransform(robot_base_frame, object_frame.header.frame_id, time, rospy.Duration(2.0))
            camera_in_base = self.tf_listener.asMatrix(robot_base_frame, object_frame.header)            
            object_in_camera = pm.toMatrix(pm.fromMsg(object_frame.pose))
            object_in_base = camera_in_base.dot(object_in_camera)

            pre_grasp_pose_in_base = None  
                        
            #get grasp type
            self.grasp_type = req.grasp_type
            if self.grasp_type == 'UseHeuristics':
                #get camera in ifco frame to be able to get object in ifco frame
                camera_in_ifco = inv(ifco_in_base).dot(camera_in_base)
                camera_in_ifco_msg = pm.toMsg(pm.fromMatrix(camera_in_ifco))

                #get pre grasp transforms in object frame for both grasp type 
                SG_pre_grasp_transform, WG_pre_grasp_transform = get_pre_grasp_transforms(self.handarm_params, self.object_type)

                SG_pre_grasp_in_object_frame_msg = pm.toMsg(pm.fromMatrix(SG_pre_grasp_transform))
                WG_pre_grasp_in_object_frame_msg = pm.toMsg(pm.fromMatrix(WG_pre_grasp_transform))
                ifco_in_base_msg = pm.toMsg(pm.fromMatrix(ifco_in_base))

                call_heuristic = rospy.ServiceProxy('target_selection', target_selection_srv.TargetSelection)
                res = call_heuristic(objectList, camera_in_ifco_msg, SG_pre_grasp_in_object_frame_msg, WG_pre_grasp_in_object_frame_msg, ifco_in_base_msg)

                pre_grasp_pose_in_ifco_frame = pm.toMatrix(pm.fromMsg(res.pre_grasp_pose_in_ifco_frame))
                target_pose_in_ifco_frame = pm.toMatrix(pm.fromMsg(res.target_pose_in_ifco_frame))

                object_in_base = ifco_in_base.dot(target_pose_in_ifco_frame)
                pre_grasp_pose_in_base = ifco_in_base.dot(pre_grasp_pose_in_ifco_frame)

                if res.grasp_type == 's':
                    self.grasp_type = 'SurfaceGrasp'
                    wall_id = 'NoWall'
                else:
                    self.grasp_type = 'WallGrasp'
                    wall_id = 'wall' + res.grasp_type[1]
                print("GRASP HEURISTICS " + self.grasp_type + " " + wall_id)
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
        ha, self.rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, object_in_base, bounding_box,
                                                                self.handarm_params, self.object_type, wall_id, ifco_in_base, req.handarm_type, pre_grasp_pose_in_base)
                                                
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
def hybrid_automaton_from_motion_sequence(motion_sequence, graph, T_object_in_base, bounding_box, handarm_params, object_type, wall_id, ifco_in_base, handarm_type, pre_grasp_pose_in_base = None):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))

    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label

    print("Creating hybrid automaton for object {} and grasp type {}.".format(object_type, grasp_type))
    if grasp_type == 'WallGrasp':
        wall_frame = get_wall_tf(ifco_in_base, wall_id)
        grasping_recipe, rviz_frames = get_hand_recipes(handarm_type).create_wall_grasp(T_object_in_base, bounding_box, wall_frame, handarm_params, object_type, ifco_in_base, pre_grasp_pose_in_base)        
        
    elif grasp_type == 'SurfaceGrasp':
        grasping_recipe, rviz_frames = get_hand_recipes(handarm_type).create_surface_grasp(T_object_in_base, bounding_box, handarm_params, object_type, ifco_in_base, pre_grasp_pose_in_base)

    return cookbook.sequence_of_modes_and_switches_with_safety_features(grasping_recipe + get_transport_recipe(handarm_params, handarm_type)), rviz_frames

def get_transport_recipe(handarm_params, handarm_type):

    lift_time = handarm_params['lift_duration']
    up_IFCO_speed = handarm_params['up_IFCO_speed']

    place_time = handarm_params['place_duration']
    down_tote_speed = handarm_params['down_tote_speed']

    # Up speed is also positive because it is defined on the world frame
    up_IFCO_twist = tra.translation_matrix([0, 0, up_IFCO_speed]);
    # Down speed is negative because it is defined on the world frame
    down_tote_twist = tra.translation_matrix([0, 0, -down_tote_speed]);    

    # assemble controller sequence
    control_sequence = []

    # 1. Lift upwards
    control_sequence.append(ha.InterpolatedHTransformControlMode(up_IFCO_twist, controller_name = 'GoUpHTransform', name = 'GoUp', goal_is_relative='1', reference_frame="world"))
 
    # 1b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoUp', 'Preplacement2', duration = lift_time))

    # # 2. Go to Preplacement position and keeping the orientation
    # control_sequence.append(ha.SlerpControlMode(handarm_params['pre_placement_pose'], controller_name = 'GoAbovePlacement', goal_is_relative='0', name = 'Preplacement1', orientation_or_and_position = 'POSITION'))
    
    # # 2b. Switch after a certain amount of time, the duration is short because the actual transition is done by the controller by exiting the infinite loop
    # control_sequence.append(ha.TimeSwitch('Preplacement1', 'Preplacement2', duration = 0.5)) 

    # 3. Change the orientation to have the hand facing the Delivery tote
    control_sequence.append(ha.SlerpControlMode(handarm_params['pre_placement_pose'], controller_name = 'GoAbovePlacement', goal_is_relative='0', name = 'Preplacement2', orientation_or_and_position = 'BOTH'))

    # 3b. Switch after a certain amount of time, the duration is short because the actual transition is done by the controller by exiting the infinite loop
    control_sequence.append(ha.TimeSwitch('Preplacement2', 'GoDown2', duration = 0.5))

    # 4. Go Down
    control_sequence.append(ha.InterpolatedHTransformControlMode(down_tote_twist, controller_name = 'GoToDropOff', name = 'GoDown2', goal_is_relative='1', reference_frame="world"))
 
    # 4b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoDown2', 'softhand_open', duration = place_time))

    # 5. Release SKU
    if handarm_type == "ClashHandKUKA":
        speed = np.array([30]) 
        thumb_pos = np.array([0, -20, 0])
        diff_pos = np.array([-10, -10, 0])
        thumb_pretension = np.array([0]) 
        diff_pretension = np.array([0]) 
        mode = np.array([0])

        thumb_contact_force = np.array([0]) 
        thumb_grasp_force = np.array([0]) 
        diff_contact_force = np.array([0]) 
        diff_grasp_force = np.array([0])    
        
        force_feedback_ratio = np.array([0]) 
        prox_level = np.array([0]) 
        touch_level = np.array([0]) 
        command_count = np.array([2]) 

        control_sequence.append(ha.ros_CLASHhandControlMode(goal = np.concatenate((speed, thumb_pos, diff_pos, thumb_contact_force, 
                                                                                thumb_grasp_force, diff_contact_force, diff_grasp_force, 
                                                                                thumb_pretension, diff_pretension, force_feedback_ratio, 
                                                                                prox_level, touch_level, mode, command_count)), name  = 'softhand_open'))
        
    else:
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = 1))

    # 5b. Switch when hand opening time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'finished', duration = handarm_params['hand_opening_duration']))

    # 6. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name  = 'finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller( name = 'JointSpaceController', type = 'InterpolatedJointController', goal  = np.zeros(7),
                                   goal_is_relative = 1, v_max = '[0,0]', a_max = '[0,0]'))
    finishedMode.set(finishedSet)  
    control_sequence.append(finishedMode)

    return control_sequence

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
