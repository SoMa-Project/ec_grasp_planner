#!/usr/bin/env python

import math
import numpy as np
from tf import transformations as tra

class BaseHandArm(dict):
    def __init__(self):
        self['mesh_file'] = "Unknown"
        self['mesh_file_scale'] = 1.
        
        self['wall_grasp'] = {}        
        self['edge_grasp'] = {}
        self['surface_grasp'] = {}
        self['wall_grasp']['object'] = {}
        self['edge_grasp']['object'] = {}
        self['surface_grasp']['object'] = {}
        self['wall_grasp']['punet'] = {}
        self['edge_grasp']['punet'] = {}
        self['surface_grasp']['punet'] = {}

class RBOHand2(BaseHandArm):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1


# This map defines all grasp parameter such as poses and configurations for a specific robot system
# Define this map for your system if you want to port the planner

#Rbo hand 2 (Ocado version with long fingers and rotated palm) mounted on WAM.
class RBOHandO2WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHandO2WAM, self).__init__()

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        # transformation between object frame and hand palm frame
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0, -0.05, 0]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        # above the object, in hand palm frame
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, -0.3]), tra.rotation_matrix(math.radians(30.0), [0, 1, 0]))

        # at grasp position, in hand palm frame
        self['surface_grasp']['object']['grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0.0, 0.05]),
                                                                                 tra.rotation_matrix(math.radians(30.0),
                                                                                                     [0, 1, 0]))

        # first motion after grasp, in hand palm frame
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, 0.]),
            tra.rotation_matrix(math.radians(-20.),
                                [0, 1, 0]))


        # second motion after grasp, in hand palm frame
        self['surface_grasp']['object']['go_up_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0, -0.3]),
                                                                            tra.rotation_matrix(math.radians(-20.),
                                                                                                [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['surface_grasp']['object']['downward_force'] = 10.

        #drop configuration - this is system specific!
        self['surface_grasp']['object']['drop_off_config'] = np.array(
            [0.600302, 0.690255, 0.00661675, 2.08453, -0.0533508, -0.267344, 0.626538])

        #synergy type for soft hand closing
        self['surface_grasp']['object']['hand_closing_synergy'] = 1

        #time of soft hand closing
        self['surface_grasp']['object']['hand_closing_duration'] = 5


        #####################################################################################
        #below are parameters for wall and edge grasp - caution: currently not working!
        #####################################################################################
        self['wall_grasp']['object']['initial_goal'] = np.array(
            [0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])
        self['wall_grasp']['object']['pregrasp_pose'] = tra.translation_matrix([-0.2, 0, 0])
        self['wall_grasp']['object']['hand_object_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(-69.0), [1, 0, 0]),
                                                                                    tra.euler_matrix(0, math.pi / 2.,
                                                                                                     math.pi / 2.))
        self['wall_grasp']['object']['grasp_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.]),
                                                                              tra.rotation_matrix(math.radians(-69.0),
                                                                                                  [1, 0, 0]),
                                                                              tra.euler_matrix(0, math.pi / 2.,
                                                                                               math.pi / 2.))
        self['wall_grasp']['object']['postgrasp_pose'] = tra.translation_matrix([-0.1, 0, -0.1])
        self['wall_grasp']['object']['table_force'] = 3.0
        self['wall_grasp']['object']['sliding_speed'] = 0.04
        self['wall_grasp']['object']['up_speed'] = 0.1
        self['wall_grasp']['object']['down_speed'] = 0.1
        self['wall_grasp']['object']['angle_of_attack'] = 1.0
        self['wall_grasp']['object']['object_lift_time'] = 1.0
        self['wall_grasp']['object']['wall_force'] = -11.0
        self['wall_grasp']['object']['valve_pattern'] = (np.array([[1, 0]] * 6), np.array([[0, 2.5]] * 6))

        self['edge_grasp']['object']['initial_goal'] = np.array(
            [0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])
        self['edge_grasp']['object']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.3])
        self['edge_grasp']['object']['hand_object_pose'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, 0.05]), tra.rotation_matrix(math.radians(10.), [1, 0, 0]),
            tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['object']['grasp_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, -0.05, 0]),
                                                                              tra.rotation_matrix(math.radians(10.),
                                                                                                  [1, 0, 0]),
                                                                              tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['object']['postgrasp_pose'] = tra.translation_matrix([0, 0, -0.1])
        self['edge_grasp']['object']['downward_force'] = 4.0
        self['edge_grasp']['object']['sliding_speed'] = 0.04
        self['edge_grasp']['object']['valve_pattern'] = (
        np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]]), np.array([[0, 3.0]] * 6))


class RBOHand2Kuka(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2Kuka, self).__init__()

        self['surface_grasp']['initial_goal'] = np.array([-0.05864322834179703, 0.4118988657714642, -0.05864200146127985, -1.6887810963180838, -0.11728653060066829, -0.8237944986945402, 0])
        self['surface_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.2])
        self['surface_grasp']['hand_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(0.), [0, 0, 1]))
        self['surface_grasp']['downward_force'] = 7.
        self['surface_grasp']['valve_pattern'] = (np.array([[ 0. ,  4.1], [ 0. ,  0.1], [ 0. ,  5. ], [ 0. ,  5.], [ 0. ,  2.], [ 0. ,  3.5]]), np.array([[1,0]]*6))
        
        self['wall_grasp']['pregrasp_pose'] = tra.translation_matrix([0.05, 0, -0.2])
        self['wall_grasp']['table_force'] = 7.0
        self['wall_grasp']['sliding_speed'] = 0.1
        self['wall_grasp']['up_speed'] = 0.1
        self['wall_grasp']['down_speed'] = 0.1
        self['wall_grasp']['wall_force'] = 10.0
        self['wall_grasp']['angle_of_attack'] = 1.0 #radians
        self['wall_grasp']['object_lift_time'] = 4.5
        
        self['edge_grasp']['initial_goal'] = np.array([-0.05864322834179703, 0.4118988657714642, -0.05864200146127985, -1.6887810963180838, -0.11728653060066829, -0.8237944986945402, 0])
        self['edge_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.3])
        self['edge_grasp']['hand_object_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.05]), tra.rotation_matrix(math.radians(10.), [1, 0, 0]), tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['grasp_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, -0.05, 0]), tra.rotation_matrix(math.radians(10.), [1, 0, 0]), tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['postgrasp_pose'] = tra.translation_matrix([0, 0, -0.1])
        self['edge_grasp']['downward_force'] = 4.0
        self['edge_grasp']['sliding_speed'] = 0.04
        self['edge_grasp']['valve_pattern'] = (np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), np.array([[0, 3.0]]*6))
        
