#!/usr/bin/env python

import math
import numpy as np
from tf import transformations as tra

# python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp wall_grasp wall_chewinggum
# python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp edge_grasp --edgedistance -0.007 edge_chewinggum/
# python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp surface_grasp test_folder

class BaseHandArm(dict):
    def __init__(self):
        self['mesh_file'] = "Unknown"
        self['mesh_file_scale'] = 1.
        
        self['wall_grasp'] = {}
        self['edge_grasp'] = {}
        self['surface_grasp'] = {}

class RBOHand2(BaseHandArm):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1


class RBOHand2WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHandP24WAM, self).__init__()

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        # transformation between object frame and hand palm frame
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        #self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, -0.05, 0.3]),tra.concatenate_matrices(tra.rotation_matrix(math.radians(90.), [0, 0, 1]),tra.rotation_matrix(math.radians(180.), [1, 0, 0])))
        self['surface_grasp']['object']['hand_transform']=tra.concatenate_matrices(tra.translation_matrix([-0.03, 0.0, 0.25]),tra.rotation_matrix(math.radians(30.0),[0, 1, 0]))

        # above the object, in hand palm frame
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0, 0.0]), tra.rotation_matrix(math.radians(10.0), [0, 1, 0]))

        # at grasp position, in hand palm frame
        self['surface_grasp']['object']['grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0.0, 0.05]),
                                                                                 tra.rotation_matrix(math.radians(30.0),
                                                                                                     [0, 1, 0]))

        # first motion after grasp, in hand palm frame only rotation
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(-15.),
                                [0, 1, 0]))

        # second motion after grasp, in hand palm frame
        self['surface_grasp']['object']['go_up_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0, -0.3]),
                                                                            tra.rotation_matrix(math.radians(-20.),
                                                                                                [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['surface_grasp']['object']['downward_force'] = 7. # might be +10/-7 ??

        #drop configuration - this is system specific!
        self['surface_grasp']['object']['drop_off_config'] = np.array(
            [0.600302, 0.690255, 0.00661675, 2.08453, -0.0533508, -0.267344, 0.626538])

        #synergy type for soft hand closing
        self['surface_grasp']['object']['hand_closing_synergy'] = 1

        #time of soft hand closing
        self['surface_grasp']['object']['hand_closing_duration'] = 5

        # time of soft hand closing
        self['surface_grasp']['object']['down_speed'] = 0.35
        self['surface_grasp']['object']['up_speed'] = 0.35

        # does the robot support impedance control
        self['isForceControllerAvailable'] = True


#Rbo hand 2 (Ocado version with long fingers and rotated palm) mounted on WAM.
class RBOHandO2WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHandO2WAM, self).__init__()

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        # transformation between object frame and hand palm frame
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['surface_grasp']['object']['hand_transform'] = tra.translation_matrix([0.0, 0.0, 0.3]);
        print(self['surface_grasp']['object']['hand_transform'])

        # above the object, in hand palm frame
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, 0.0]), tra.rotation_matrix(math.radians(-30.0), [0, 1, 0]))
# this is from wallGrasp branch
# tra.translation_matrix([-0.03, 0, -0.28]), tra.rotation_matrix(math.radians(0.), [0, 1, 0]))

        # at grasp position, in hand palm frame
        self['surface_grasp']['object']['grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0.0, 0.05]),
                                                                                 tra.rotation_matrix(math.radians(30.0),
                                                                                                     [0, 1, 0]))

        # first motion after grasp, in hand palm frame only rotation
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(-15.),
                                [0, 1, 0]))


        # second motion after grasp, in hand palm frame
        self['surface_grasp']['object']['go_up_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0, -0.3]),
                                                                            tra.rotation_matrix(math.radians(-20.),
                                                                                                [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['surface_grasp']['object']['downward_force'] = 7. # might be +10/-7 ??

        #drop configuration - this is system specific!
        self['surface_grasp']['object']['drop_off_config'] = np.array(
            [0.600302, 0.690255, 0.00661675, 2.08453, -0.0533508, -0.267344, 0.626538])

        #synergy type for soft hand closing
        self['surface_grasp']['object']['hand_closing_synergy'] = 1

        #time of soft hand closing
        self['surface_grasp']['object']['hand_closing_duration'] = 5

        # time of soft hand closing
        self['surface_grasp']['object']['down_speed'] = 0.35
        self['surface_grasp']['object']['up_speed'] = 0.35

        self['isForceControllerAvailable'] = True

class RBOHand2Kuka(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2Kuka, self).__init__()

        #old parameters below - must be updated to new convention!
        self['surface_grasp']['object']['initial_goal'] = np.array([-0.05864322834179703, 0.4118988657714642, -0.05864200146127985, -1.6887810963180838, -0.11728653060066829, -0.8237944986945402, 0])
        self['surface_grasp']['object']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.2])
        self['surface_grasp']['object']['hand_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(0.), [0, 0, 1]))
        self['surface_grasp']['object']['downward_force'] = 7.
        self['surface_grasp']['object']['valve_pattern'] = (np.array([[ 0. ,  4.1], [ 0. ,  0.1], [ 0. ,  5. ], [ 0. ,  5.], [ 0. ,  2.], [ 0. ,  3.5]]), np.array([[1,0]]*6))
        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(0.), [0, 0, 1]))
        self['surface_grasp']['object']['pregrasp_transform'] = tra.translation_matrix([0, 0, -0.2])
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.0]), tra.rotation_matrix(math.radians(-30.0), [0, 1, 0]))
        self['surface_grasp']['object']['hand_closing_duration'] = 5
        self['surface_grasp']['object']['hand_closing_synergy'] = 5
        self['surface_grasp']['object']['drop_off_config']=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        self['surface_grasp']['object']['up_speed'] = 0.1
        self['surface_grasp']['object']['down_speed'] = 0.1


        self['wall_grasp']['object']['pregrasp_pose'] = tra.translation_matrix([0.05, 0, -0.2])
        self['wall_grasp']['object']['table_force'] = 7.0
        self['wall_grasp']['object']['sliding_speed'] = 0.1
        self['wall_grasp']['object']['up_speed'] = 0.1
        self['wall_grasp']['object']['down_speed'] = 0.1
        self['wall_grasp']['object']['wall_force'] = 10.0
        self['wall_grasp']['object']['angle_of_attack'] = 1.0 #radians
        self['wall_grasp']['object']['object_lift_time'] = 4.5
        
        self['edge_grasp']['object']['initial_goal'] = np.array([-0.05864322834179703, 0.4118988657714642, -0.05864200146127985, -1.6887810963180838, -0.11728653060066829, -0.8237944986945402, 0])
        self['edge_grasp']['object']['pregrasp_pose'] = tra.translation_matrix([0.2, 0, 0.4])
        self['edge_grasp']['object']['hand_object_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.05]), tra.rotation_matrix(math.radians(10.), [1, 0, 0]), tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['object']['grasp_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, -0.05, 0]), tra.rotation_matrix(math.radians(10.), [1, 0, 0]), tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['object']['postgrasp_pose'] = tra.translation_matrix([0, 0, -0.1])
        self['edge_grasp']['object']['downward_force'] = 4.0
        self['edge_grasp']['object']['sliding_speed'] = 0.04
        self['edge_grasp']['object']['valve_pattern'] = (np.array([[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]), np.array([[0, 3.0]]*6))

class PisaKuka(Kuka, PisaHand):
    def __init__(self, **kwargs):
        super(PisaKuka,self).__init__(**kwargs)



