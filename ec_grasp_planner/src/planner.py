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

import pyddl

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')
sys.path.append(pkg_path + '/../hybrid-automaton-tools-py/')
import hatools.components as ha
import hatools.cookbook as cookbook

import handarm_parameters

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

def create_wall_grasp(object_frame, support_surface_frame, wall_frame, handarm_params):   
    downward_force = handarm_params['wall_grasp']['table_force']
    sliding_speed = handarm_params['wall_grasp']['sliding_speed']
    wall_force = handarm_params['wall_grasp']['wall_force']
    angle_of_attack = handarm_params['wall_grasp']['angle_of_attack']
    
    rot = tra.rotation_matrix(angle_of_attack, [1, 0, 0])
    curree = np.dot(object_frame, rot)
    goal = np.dot(tra.translation_matrix([0, 0, -0.3]), curree)
    
    rviz_frames = []
    goal1 = np.copy(wall_frame)
    goal1[:3,3] = tra.translation_from_matrix(object_frame)
    rviz_frames.append(goal1.dot(tra.translation_matrix([0, 0, 0.2])).dot(rot))
    rviz_frames.append(goal1.dot(rot))
    rviz_frames.append(np.dot(wall_frame, rot))
        
    control_sequence = []
    
    control_sequence.append(ha.HTransformControlMode(curree, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.02'))
    control_sequence.append(ha.HTransformControlMode(goal, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([0, 0, downward_force, 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.ControlMode('').set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.25, desired_displacement=tra.translation_matrix([sliding_speed, 0, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0])))))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([wall_force, 0, 0, 0, 0, 0]), norm_weights = np.array([1, 0, 0, 0, 0, 0]), jump_criterion = "THRESH_LOWER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.GravityCompensationMode())
    hand_closing_time = 2.5
    control_sequence[-1].controlset.add(ha.RBOHandController(goal = np.array([[1,0]]*6), valve_times = np.array([[0,hand_closing_time]]*6), goal_is_relative = '1'))
    control_sequence.append(ha.TimeSwitch('', '', duration = 1. + hand_closing_time))
    control_sequence.append(ha.HTransformControlMode(curree, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.01'))
    control_sequence.append(ha.GravityCompensationMode())
    
    return cookbook.sequence_of_modes_and_switches(control_sequence), rviz_frames

def create_edge_grasp(object_frame, support_surface_frame, edge_frame, handarm_params):
    edge_distance_factor = handarm_params['edge_grasp']['edge_distance_factor']
    distance = handarm_params['edge_grasp']['distance']
    downward_force = handarm_params['edge_grasp']['downward_force']
    sliding_speed = handarm_params['edge_grasp']['sliding_speed']
    angle_of_sliding = handarm_params['edge_grasp']['angle_of_sliding']

    initial_cspace_goal = np.array([0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])

    curree = object_frame # TODO 
    curree = np.dot(curree, tra.rotation_matrix(angle_of_sliding, [1, 0, 0]))
    goal = np.dot(tra.translation_matrix([0, 0, -0.3]), curree)
    hand_closing_time = 3.0
    
    rviz_frames = []
    rot = tra.rotation_matrix(angle_of_sliding, [1, 0, 0])
    goal1 = np.copy(edge_frame)
    goal1[:3,3] = tra.translation_from_matrix(object_frame)
    rviz_frames.append(goal1.dot(tra.translation_matrix([0, 0, -0.2])).dot(rot))
    rviz_frames.append(goal1.dot(rot))
    rviz_frames.append(edge_frame.dot(rot))
    
    control_sequence = []
    control_sequence.append(ha.HTransformControlMode(curree, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.001'))
    control_sequence.append(ha.HTransformControlMode(goal, controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([0, 0, downward_force, 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.ControlMode('').set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.25, desired_displacement=tra.translation_matrix([0, -sliding_speed, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0])))))
    control_sequence.append(ha.FrameDisplacementSwitch('', '', epsilon = str(edge_distance_factor + distance), negate = '1', goal = np.array([0, 0, 0]), goal_is_relative = '1', jump_criterion = "NORM_L2", frame_id = 'EE'))
    control_sequence.append(ha.GravityCompensationMode())
    control_sequence[-1].controlset.add(ha.RBOHandController(goal = np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), valve_times = np.array([[0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time], [0,hand_closing_time]]), goal_is_relative = '1'))
    control_sequence.append(ha.TimeSwitch('', '', duration = 1.0 + hand_closing_time))
    control_sequence.append(ha.HTransformControlMode(tra.translation_matrix([0, 0, -0.1]), controller_name = 'GoToCartesianConfig', goal_is_relative='1'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.001', goal_is_relative = '1'))
    control_sequence.append(ha.GravityCompensationMode())
    
    return cookbook.sequence_of_modes_and_switches(control_sequence), rviz_frames

def create_surface_grasp(object_frame, support_surface_frame, handarm_params):
    initial_cspace_goal = handarm_params['surface_grasp']['initial_goal']
    pregrasp_pose = handarm_params['surface_grasp']['pregrasp_pose']
    hand_pose = handarm_params['surface_grasp']['pose']
    downward_force = handarm_params['surface_grasp']['downward_force']
    valve_pattern = handarm_params['surface_grasp']['valve_pattern']
    hand_closing_time = np.max(valve_pattern) + 1.
    
    rviz_frames = []
    goal = np.copy(support_surface_frame)
    goal[:3,3] = tra.translation_from_matrix(object_frame)
    rviz_frames.append(goal.dot(pregrasp_pose).dot(hand_pose))
    rviz_frames.append(goal.dot(tra.translation_matrix([0, 0, 0.1])).dot(hand_pose))
    
    goals = rviz_frames
    
    control_sequence = []
    control_sequence.append(ha.JointControlMode(initial_cspace_goal, controller_name = 'GoToJointConfig'))
    control_sequence.append(ha.JointConfigurationSwitch('', '', controller = 'GoToJointConfig', epsilon = str(math.radians(7.))))
    control_sequence.append(ha.HTransformControlMode(np.hstack(goals), controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([0, 0, downward_force, 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1'))
    control_sequence.append(ha.ControlMode('').set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.0, desired_displacement=tra.translation_matrix([0, 0, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0])))))
    #control_sequence[-1].controlset.add(ha.RBOHandController(goal = valve_pattern[0], valve_times = valve_pattern[1], goal_is_relative = '1'))
    control_sequence.append(ha.TimeSwitch('', '', duration = hand_closing_time))
    control_sequence.append(ha.HTransformControlMode(goals[0], controller_name = 'GoToCartesianConfig', goal_is_relative='0'))
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoToCartesianConfig', epsilon = '0.001'))
    control_sequence.append(ha.GravityCompensationMode())
            
    return cookbook.sequence_of_modes_and_switches(control_sequence), rviz_frames

def transform_msg_to_homogenous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]), tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))

def homogenous_tf_to_pose_msg(htf):
    return Pose(position = Point(*tra.translation_from_matrix(htf).tolist()), orientation = Quaternion(*tra.quaternion_from_matrix(htf).tolist()))


def get_node_from_actions(actions, action_name, graph):
    return graph.nodes[[int(m.sig[1][1:]) for m in actions if m.name == action_name][0]]

def hybrid_automaton_from_motion_sequence(motion_sequence, graph, T_robot_base_frame, T_object_in_base, handarm_params):
    assert(len(motion_sequence) > 1)
    assert(motion_sequence[-1].name.startswith('grasp'))
    
    grasp_type = graph.nodes[int(motion_sequence[-1].sig[1][1:])].label
    #grasp_frame = grasp_frames[grasp_type]
    
    if grasp_type == 'EdgeGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        edge_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        edge_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(edge_frame_node.transform))
        return create_edge_grasp(T_object_in_base, support_surface_frame, edge_frame, handarm_params)
    elif grasp_type == 'WallGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'move_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        wall_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        wall_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(wall_frame_node.transform))
        return create_wall_grasp(T_object_in_base, support_surface_frame, wall_frame, handarm_params)
    elif grasp_type == 'SurfaceGrasp':
        support_surface_frame_node = get_node_from_actions(motion_sequence, 'grasp_object', graph)
        support_surface_frame = T_robot_base_frame.dot(transform_msg_to_homogenous_tf(support_surface_frame_node.transform))
        return create_surface_grasp(T_object_in_base, support_surface_frame, handarm_params)
    else:
        raise "Unknown grasp type: ", grasp_type

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

def publish_rviz_markers(frames, frame_id, handarm_params):
    from visualization_msgs.msg import MarkerArray
    from visualization_msgs.msg import Marker
    marker_pub = rospy.Publisher('planned_grasp_path', MarkerArray, queue_size=1, latch=False)
    markers = MarkerArray()
    
    timestamp = rospy.Time.now()
    
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
        
        markers.markers.append(msg)
    
    for f1, f2 in zip(frames, frames[1:]):
        msg = Marker()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.frame_locked = True # False
        msg.id = markers.markers[-1].id + 1
        msg.action = Marker.ADD
        msg.lifetime = rospy.Duration()
        msg.type = Marker.ARROW
        msg.color.g = msg.color.b = 0
        msg.color.r = msg.color.a = 1
        msg.scale.x = 0.01 # shaft diameter
        msg.scale.y = 0.03 # head diameter
        msg.points.append(homogenous_tf_to_pose_msg(f1).position)
        msg.points.append(homogenous_tf_to_pose_msg(f2).position)
        
        markers.markers.append(msg)
    
    br = tf.TransformBroadcaster()

    r = rospy.Rate(5);
    while not rospy.is_shutdown():
        marker_pub.publish(markers)
        
        for i, f in enumerate(frames):
            br.sendTransform(tra.translation_from_matrix(f),
                        tra.quaternion_from_matrix(f),
                        rospy.Time.now(),
                        "dgb_frame_" + str(i),
                        frame_id)
        
        r.sleep()


def main(**args):
    rospy.init_node('ec_planner')
    
    handarm_params = handarm_parameters.__dict__[args['handarm']]()
    
    tf_listener = tf.TransformListener()
    
    robot_base_frame = args['robot_base_frame']
    object_frame = args['object_frame']
    
    # make sure those frames exist and we can transform between them
    tf_listener.waitForTransform(object_frame, robot_base_frame, rospy.Time(), rospy.Duration(10.0))
    
    #p.header.stamp = rospy.Time()
    #p_robotbaseframe = tf_listener.transformPose(robot_base_frame, p)
    #t_robotbaseframe = get_numpy_matrix(self.li, p_robotbaseframe.pose)

    # get grasp from graph representation
    grasp_path = None
    while grasp_path is None:
        # get geometry graph
        graph = rospy.wait_for_message('geometry_graph', Graph)
        
        graph.header.stamp = rospy.Time.now() + rospy.Duration(0.5)
        
        tf_listener.waitForTransform(robot_base_frame, graph.header.frame_id, graph.header.stamp, rospy.Duration(10.0))
        graph_in_base = tf_listener.asMatrix(robot_base_frame, graph.header)
        
        tf_listener.waitForTransform(robot_base_frame, object_frame, graph.header.stamp, rospy.Duration(10.0))
        object_in_base = tf_listener.asMatrix(robot_base_frame, Header(0, rospy.Time(), object_frame))
        
        print("Received graph with {} nodes and {} edges.".format(len(graph.nodes), len(graph.edges)))
        
        hand_node_id = [n.label for n in graph.nodes].index("Positioning")
        object_node_id = [n.label for n in graph.nodes].index("Slide")
        grasp_path = find_a_path(hand_node_id, object_node_id, graph, args['grasp'], verbose = True)
        
        # identify potential goal nodes (based on grasp type)
        #goal_node_ids = [i for i, n in enumerate(graph.nodes) if n.label in args['grasp']]
        #if len(goal_node_ids) > 0:
            # get all paths that end in goal_ids
            #paths = find_all_paths(0, goal_node_ids, graph)
            #if len(paths) > 0:
            #    grasp_path = paths[0]
            #    break
            #grasp_path = find_a_path(hand_start_node_id, object_start_node_id, graph, goal_node_id = None, verbose = False):
        
        rospy.sleep(0.3)
    
    # Turn grasp into hybrid automaton
    ha, rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, graph_in_base, object_in_base, handarm_params)
    
    # call a service
    if args['ros_service_call']:
        call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        call_ha(ha.xml())
    
    # write to file
    if args['file_output']:
        with open('hybrid_automaton.xml', 'w') as outfile:
            outfile.write(ha.xml())
    
    if args['rviz']:
        print "Press Cntrl-C to stop sending visualization_msgs/MarkerArray on topic '/planned_grasp_path' ..."
        publish_rviz_markers(rviz_frames, robot_base_frame, handarm_params)
        #rospy.spin()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn path in graph into hybrid automaton.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--ros_service_call', action='store_true', default = False,
                        help='Whether to send the hybrid automaton to a ROS service called /update_hybrid_automaton.')
    parser.add_argument('--file_output', action='store_true', default = False,
                        help='Whether to write the hybrid automaton to a file called hybrid_automaton.xml.')
    grasp_choices = ["any", "EdgeGrasp", "WallGrasp", "SurfaceGrasp"]
    parser.add_argument('--grasp', choices=grasp_choices, default=grasp_choices[0],
                        help='Which grasp type to use.')
    parser.add_argument('--grasp_id', type=int, default=-1,
                        help='Which specific grasp to use. Ignores any values < 0.')
    parser.add_argument('--rviz', action='store_true', default = False,
                        help='Whether to send marker messages that can be seen in RViz and represent the chosen grasping motion.')
    parser.add_argument('--robot_base_frame', type=str, default = 'world',
                        help='Name of the robot base frame.')
    parser.add_argument('--object_frame', type=str, default = 'object',
                        help='Name of the object frame.')
    parser.add_argument('--handarm', type=str, default = 'RBOHand2WAM',
                        help='Python class that contains configuration parameters for hand and arm-specific properties.')
    
    args = parser.parse_args()
    
    if args.grasp == 'any':
        args.grasp = grasp_choices[1:]
    else:
        args.grasp = [args.grasp]
    
    if args.grasp_id >= 0:
        tmp = [g + '_' + str(args.grasp_id) for g in args.grasp]
        args.grasp = tmp
    
    main(**vars(args))
