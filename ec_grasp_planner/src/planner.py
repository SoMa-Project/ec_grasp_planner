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
        self.tf_listener.waitForTransform(object_frame, robot_base_frame, rospy.Time(), rospy.Duration(10.0))

        # --------------------------------------------------------
        # Get grasp from graph representation
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
            
            print(hand_node_id)
            print(object_node_id)
            print(self.grasp_type)
  
            grasp_path = find_a_path(hand_node_id, object_node_id, graph, self.grasp_type, verbose=True)

            rospy.sleep(0.3)

        # --------------------------------------------------------
        # Turn grasp into hybrid automaton
        ha, self.rviz_frames = hybrid_automaton_from_motion_sequence(grasp_path, graph, graph_in_base, object_in_base,
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



def create_wall_grasp(object_frame, support_surface_frame, wall_frame, handarm_params, object_type):
    # Get the relevant parameters
    print(object_type)
    if (object_type in handarm_params['wall_grasp']):            
        params = handarm_params['wall_grasp'][object_type]
    else:
        params = handarm_params['wall_grasp']['object']
        

    # Get the parameters from the handarm_parameters.py file
    table_force = params['table_force']
    sliding_speed = params['sliding_speed']
    up_speed = params['up_speed']
    down_speed = params['down_speed']
    wall_force = params['wall_force']
    angle_of_attack = params['angle_of_attack']
    pregrasp_pose = params['pregrasp_pose']
    object_lift_time = params['object_lift_time']

    # Get the pose above the object
    rviz_frames = []
    ssf = np.copy(support_surface_frame)
    ssf[:3,3] = tra.translation_from_matrix(object_frame)
    goal_ = (ssf.dot(pregrasp_pose))
    rot = tra.rotation_matrix(-1.57, [0, 0, 1])
    goal__ = (goal_.dot(rot))
    rot = tra.rotation_matrix(3.14, [1, 0, 0])
    goal___ = (goal__.dot(rot))
    rot = tra.rotation_matrix(angle_of_attack, [0, 1, 0])
    goalr = (goal___.dot(rot))
    rviz_frames.append(goalr)
    goal = (tra.translation_matrix([0, -0.2, 0]).dot(goalr))
    rviz_frames.append(goal)

    # Create the directions to go up/down and towards the wall
    dirDown = tra.translation_matrix([0, 0, -down_speed]);
    dirWall = tra.translation_matrix([0, sliding_speed, 0]);
    dirUp = tra.translation_matrix([0, 0, up_speed]);

    control_sequence = []

    # 1. Go above the object
    control_sequence.append(ha.HTransformControlMode(goal, controller_name = 'GoAboveObject', goal_is_relative='0'))

    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('', '', controller = 'GoAboveObject', epsilon = '0.01'))

    # 2. Go down onto the object
    control_sequence.append(ha.HTransformControlMode(dirDown, controller_name = 'GoDown', goal_is_relative='1'))

    # 2b. Switch when the f/t sensor is triggered with normal force from the table
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([table_force, 0, 0, 0, 0, 0]),
        norm_weights = np.array([1, 0, 0, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1'))

    # 3. Lift upwards
    control_sequence.append(ha.HTransformControlMode(dirUp, controller_name = 'Lift1', goal_is_relative='1'))

    # 3b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('', '', duration = 1.5))

    # 4. Go towards the wall
    control_sequence.append(ha.HTransformControlMode(dirWall, controller_name = 'SlideToWall', goal_is_relative='1'))

    # 4b. Switch when the f/t sensor is triggered with normal force from the table
    control_sequence.append(ha.ForceTorqueSwitch('', '', goal = np.array([0, 0, wall_force, 0, 0, 0]),
        norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", goal_is_relative = '1'))

    # 5. Lift upwards
    control_sequence.append(ha.HTransformControlMode(dirUp, controller_name = 'Lift2', goal_is_relative='1'))

    # 5b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('', '', duration = object_lift_time))

    # 6. Go to gravity compensation mode to finish
    control_sequence.append(ha.GravityCompensationMode())    

    return cookbook.sequence_of_modes_and_switches(control_sequence), rviz_frames

# ================================================================================================
def create_surface_grasp(object_frame, support_surface_frame, handarm_params, object_type):

    # Get the relevant parameters for hand object combination
    print(object_type)
    if (object_type in handarm_params['surface_grasp']):            
        params = handarm_params['surface_grasp'][object_type]
    else:
        params = handarm_params['surface_grasp']['object']
        

    hand_transform = params['hand_transform']
    pregrasp_transform = params['pregrasp_transform']
    grasp_transform = params['grasp_transform']
    post_grasp_rotation= params['post_grasp_transform'] # TODO: USE THIS!!!
    go_up_transform = params['go_up_tranform']

    drop_off_config = params['drop_off_config']

    downward_force = params['downward_force']
    hand_closing_time = params['hand_closing_duration']
    hand_synergy = params['hand_closing_synergy']

    # Set the initial pose above the object
    #goal_ = np.copy(support_surface_frame)

    goal_ = np.copy(object_frame) #TODO: this should be support_surface_frame
    goal_[:3,3] = tra.translation_from_matrix(object_frame)
    goal_ =  goal_.dot(hand_transform)

    #the grasp frame is symmetrical - check which side is nicer to reach
    #this is a hacky first version for our WAM
    zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    if goal_[0][0]<0:
        goal_ = goal_.dot(zflip_transform)


    pre_grasp_pose = goal_.dot(pregrasp_transform)
    grasp_pose = goal_.dot(grasp_transform)
    go_up_pose = goal_.dot(go_up_transform)

    # Set the frames to visualize with RViz
    rviz_frames = []
    rviz_frames.append(pre_grasp_pose)
    rviz_frames.append(grasp_pose)
    rviz_frames.append(go_up_pose)

    # assemble controller sequence
    control_sequence = []

    # 1. Go above the object - Pregrasp    
    control_sequence.append(ha.HTransformControlMode(pre_grasp_pose, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'Pregrasp'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Pregrasp', 'GoDown', controller = 'GoAboveObject', epsilon = '0.01'))
 
    # 2. Go down onto the object Godown
    control_sequence.append(ha.HTransformControlMode(grasp_pose, controller_name = 'GoDown', goal_is_relative='0', name = 'GoDown'))
 
    # 2b. Switch when the f/t sensor is triggered with normal force from the table 
    # rotate force reading into grasp pose frame
    # todo: use the new ft sensor in world frame
    force  = np.array([0, 0, downward_force, 0])
    force  =  grasp_pose.dot(force) # in the EE frame
    force.resize(6)

    control_sequence.append(ha.ForceTorqueSwitch('GoDown', 'softhand_close', 'ForceSwitch', goal = force,
        norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_LOWER_BOUND", goal_is_relative = '1'))
 
    # 3. Preserve the position
    desired_displacement = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    force_gradient = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0 ], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])    
    
    if params.isForceControllerAvailable:
        control_sequence.append(ha.HandControlMode_ForceHT(name  = 'softhand_close', synergy = hand_synergy,
                                                        desired_displacement = desired_displacement, 
                                                        force_gradient = force_gradient, 
                                                        desired_force_dimension = desired_force_dimension))
    else:
        control_sequence.append(ha.close_rbohand())

    # 3b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration = hand_closing_time))
# 
    # 4. Lift upwards
    control_sequence.append(ha.HTransformControlMode(go_up_pose, controller_name = 'GoUpHTransform', name = 'GoUp', goal_is_relative='0' ))
 
    # 4b. Switch when joint is reached
    control_sequence.append(ha.FramePoseSwitch('GoUp', 'GoDropOff', controller = 'GoUpHTransform', epsilon = '0.01'))
     
    # 5. Go to dropOFF 
    control_sequence.append(ha.JointControlMode(drop_off_config, controller_name = 'GoToDropJointConfig', name = 'GoDropOff'))
 
    # 5.b  Switch when joint is reached
    control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'finished', controller = 'GoToDropJointConfig', epsilon = str(math.radians(7.))))    
 
    # 6. Block joints to finish motion and hold object in air
    finishedMode = ha.ControlMode(name  = 'finished')
    finishedSet = ha.ControlSet()
    finishedSet.add(ha.Controller( name = 'JointSpaceController', type = 'JointController', goal  = np.zeros(7),
                                   goal_is_relative = 1, v_max = '[0,0]', a_max = '[0,0]'))
    finishedMode.set(finishedSet)  
    control_sequence.append(finishedMode)    
    

    return cookbook.sequence_of_modes_and_switches_with_saftyOn(control_sequence), rviz_frames

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
