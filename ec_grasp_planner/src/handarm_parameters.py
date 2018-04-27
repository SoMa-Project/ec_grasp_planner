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

        # strategy types
        self['wall_grasp'] = {}        
        self['edge_grasp'] = {}
        self['surface_grasp'] = {}

        # surface grasp parameters for differnt objects
        # 'object' is the default parameter set
        self['surface_grasp']['object'] = {}
        self['surface_grasp']['punet'] = {}

        # wall grasp parameters for differnt objects
        self['wall_grasp']['object'] = {}
        self['wall_grasp']['object'] = {}

        # wall grasp parameters for differnt objects
        self['edge_grasp']['object'] = {}
        self['edge_grasp']['object'] = {}


        self['isForceControllerAvailable'] = False


class RBOHand2(BaseHandArm):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1


# This map defines all grasp parameter such as poses and configurations for a specific robot system


# Define this map for your system if you want to port the planner
#Rbo hand 2 (P24 fingers and rotated palm) mounted on WAM.
class RBOHandP24WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHandP24WAM, self).__init__()

        # does the robot support impedance control
        self['isForceControllerAvailable'] = True

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        # transformation (only rotation) between object frame and hand palm frame
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.3]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        # above the object, in hand palm frame
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.05, 0, 0.0]), tra.rotation_matrix(math.radians(10.0), [0, 1, 0]))

        # at grasp position, in hand palm frame
        self['surface_grasp']['object']['grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0.0, 0.05]),
                                                                                 tra.rotation_matrix(math.radians(30.0),
                                                                                                     [0, 1, 0]))

        # first motion after grasp, in hand palm frame only rotation
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(-15.),
                                [0, 1, 0]))

        # second motion after grasp, in hand palm frame
        self['surface_grasp']['object']['go_up_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.03, 0, -0.3]),
                                                                            tra.rotation_matrix(math.radians(-20.),
                                                                                                [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['surface_grasp']['object']['downward_force'] = 4 # might be +10/-7 ??

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
        self['surface_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.06])  # first value: rotational, second translational


        #####################################################################################
        # below are parameters for wall grasp with P24 fingers (standard RBO hand)
        #####################################################################################
        self['wall_grasp']['object']['hand_closing_duration'] = 5
        self['wall_grasp']['object']['initial_goal'] = np.array(
            [0.439999, 0.624437, -0.218715, 1.71695, -0.735594, 0.197093, -0.920799])

        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['wall_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(180.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(90.0), [0, 0, 1]),
            ))

        # the pre-approach pose should be:
        # - floating above and behind the object,
        # - fingers pointing downwards
        # - palm facing the object and wall
        self['wall_grasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.23, 0, -0.15]), #23 cm above object, 15 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        math.radians(30.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                    tra.rotation_matrix(                #this makes the fingers point downwards
                        math.radians(0.0), [0, 0, 1]),
            ))

        # first motion after grasp, in hand palm frame
        self['wall_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]), #nothing right now
            tra.rotation_matrix(math.radians(0.0),
                                [0, 1, 0]))

        # drop configuration - this is system specific!
        self['wall_grasp']['object']['drop_off_config'] = np.array(
            [0.25118, 0.649543, -0.140991, 1.79668, 0.0720235, 0.453135, -1.03957])

        self['wall_grasp']['object']['table_force'] = 1.5
        self['wall_grasp']['object']['lift_dist'] = 0.1 #short lift after initial contact (before slide)
        self['wall_grasp']['object']['sliding_dist'] = 0.4 #sliding distance, should be min. half Ifco size
        self['wall_grasp']['object']['up_dist'] = 0.2
        self['wall_grasp']['object']['down_dist'] = 0.25
        self['wall_grasp']['object']['go_down_velocity'] = np.array([0.125, 0.06]) #first value: rotational, second translational
        self['wall_grasp']['object']['slide_velocity'] = np.array([0.125, 0.06])
        self['wall_grasp']['object']['wall_force'] = 3.0

        #####################################################################################
        # below are parameters for edge grasp with P24 fingers (standard RBO hand)
        #####################################################################################
        self['edge_grasp']['object']['hand_closing_duration'] = 5
        self['edge_grasp']['object']['initial_goal'] = np.array(
            [0.163113, 0.600424, -0.072204, 1.92395, 0.0919598, 0.520367, -0.129253])

        # transformation between hand and EC frame
        #self['edge_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
        #    tra.translation_matrix([0.0, 0.0, 0.0]),
        #    tra.concatenate_matrices(
        #        tra.rotation_matrix(math.radians(90.), [0, 0, 1]),
        #        tra.rotation_matrix(math.radians(180.), [1, 0, 0])
        #    )
        #)


        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['edge_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(90.0), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(-90.0), [0, 0, 1]),
            ))

        # the pre-approach pose should be:
        # - floating above the object,
        # - fingers pointing downwards
        # - palm facing the object and wall
        self['edge_grasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, -0.23]),  # 23 cm above object
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        # first motion after grasp, in hand palm frame
        self['edge_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),  # nothing right now
            tra.rotation_matrix(math.radians(0.0),
                                [0, 1, 0]))

        # drop configuration - this is system specific!
        self['edge_grasp']['object']['drop_off_config'] = np.array(
            [0.25118, 0.649543, -0.140991, 1.79668, 0.0720235, 0.453135, -1.03957])

        self['edge_grasp']['object']['table_force'] = 1.5
        self['edge_grasp']['object']['up_dist'] = 0.2
        self['edge_grasp']['object']['down_dist'] = 0.25
        self['edge_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.06])  # first value: rotational, second translational
        self['edge_grasp']['object']['slide_velocity'] = np.array([0.125, 0.03])

class RBOHandP11WAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)

class RBOHandP24_opposableThumbWAM(RBOHand2):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)

#Rbo hand 2 (Ocado version with long fingers and rotated palm) mounted on WAM.
class RBOHandO2WAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        super(RBOHandO2WAM, self).__init__()

        # This setup cna grasp Ocado an punnet form IFCO
        # above the object, in hand palm frame
        # palm shifted back more then P24 due to increased size of fingers
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.08, 0, 0.0]), tra.rotation_matrix(math.radians(20.0), [0, 1, 0]))

        # increase rotation so fingers ar almost orthogonal to the punnet
        self['surface_grasp']['object']['grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.03, 0.0, 0.05]),
            tra.rotation_matrix(math.radians(50.0),
                                [0, 1, 0]))


#TODO: this needs to be adapted similar to match the frames above!
#The map is now 3d and the frame definitions changed.
class RBOHand2Kuka(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2Kuka, self).__init__()


        #old parameters below - must be updated to new convention!
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
