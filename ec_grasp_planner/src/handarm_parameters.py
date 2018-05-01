#!/usr/bin/env python

import math
import numpy as np
import rospy
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
        self['object'] = {}
        self['punnet'] = {}
        self['netbag'] = {}
        self['salad'] = {}

        # surface grasp parameters for different objects
        # 'object' is the default parameter set
        self['surface_grasp']['object'] = {}
        self['surface_grasp']['cucumber'] = {}
        self['surface_grasp']['punnet'] = {}
        self['surface_grasp']['netbag'] = {}

        # wall grasp parameters for different objects
        self['wall_grasp']['object'] = {}
        self['wall_grasp']['punnet'] = {}
        self['wall_grasp']['salad'] = {}
        self['wall_grasp']['netbag'] = {}
        self['wall_grasp']['mango'] = {}

        # wall grasp parameters for different objects
        self['edge_grasp']['object'] = {}


        self['isForceControllerAvailable'] = False


class RBOHand2(BaseHandArm):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1


# This map defines all grasp parameters such as poses and configurations for a specific robot system

# Define this map for your system if you want to port the planner
# Rbo hand 2 (P24 fingers and rotated palm) mounted on WAM.
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

        # This setup can grasp an Ocado punnet from IFCO
        # above the object, in hand palm frame
        # palm shifted back more than P24 due to increased size of fingers
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.08, 0, 0.0]), tra.rotation_matrix(math.radians(20.0), [0, 1, 0]))

        # increase rotation so fingers ar almost orthogonal to the punnet
        self['surface_grasp']['object']['grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.03, 0.0, 0.05]),
            tra.rotation_matrix(math.radians(50.0),
            tra.rotation_matrix(math.radians(50.0),
                                [0, 1, 0])))


class KUKA(BaseHandArm):
    def __init__(self, **kwargs):
        super(KUKA, self).__init__()

        ####################################################################################
        # General params 
        ####################################################################################

        # TRIK controller speeds
        self['down_IFCO_speed'] = 0.03
        self['up_IFCO_speed'] = 0.03
        self['down_tote_speed'] = 0.05

        self['rotate_duration'] = 3
        self['lift_duration'] = 13
        self['place_duration'] = 5

        self['pre_placement_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]), tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

        self['isForceControllerAvailable'] = False

        self['hand_closing_synergy'] = 1


        ####################################################################################
        # General vision related params 
        ####################################################################################

        self['object']['obj_bbox_uncertainty_offset'] = 0.05
        self['netbag']['obj_bbox_uncertainty_offset'] = 0.06
        self['punnet']['obj_bbox_uncertainty_offset'] = 0.08
        self['salad']['obj_bbox_uncertainty_offset'] = 0.1

        #####################################################################################
        # Common surface grasp params
        #####################################################################################

        self['surface_grasp']['object']['downward_force'] = 4.

        #####################################################################################
        # Common wall grasp params
        #####################################################################################
        
        self['wall_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(180.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(90.0), [0, 0, 1]),
            )

        self['wall_grasp']['object']['downward_force'] = 2.

        self['wall_grasp']['object']['wall_force'] = 3.5
        
        self['wall_grasp']['object']['slide_speed'] = 0.05

class RBOHandO2KUKA(KUKA):
    def __init__(self, **kwargs):
        super(RBOHandO2KUKA, self).__init__()

        ####################################################################################
        # RBO specific params irrespective of grasp type 
        ####################################################################################

        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"

        self['mesh_file_scale'] = 0.1
        
        self['hand_closing_duration'] = 6

        self['hand_opening_duration'] = 4

        self['isInPositionControl'] = True

        ####################################################################################
        # RBO specific params for surface grasp
        ####################################################################################

        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, -0.03, 0.15]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))
        self['surface_grasp']['punnet']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, -0.055, 0.15]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        self['surface_grasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])


        ####################################################################################
        # RBO specific params for wall grasp
        ####################################################################################

        self['wall_grasp']['object']['pre_approach_transform'] = tra.translation_matrix([-0.20, 0, -0.03])
        self['wall_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.005, 0, -0.01]),
                                                                 tra.rotation_matrix(math.radians(-5), [0, 1, 0]))




class PISAHandKUKA(KUKA):
    def __init__(self, **kwargs):
        super(PISAHandKUKA, self).__init__()

        ####################################################################################
        # IIT specific params irrespective of grasp type 
        ####################################################################################

        self['hand_closing_duration'] = 2

        self['IMU_closing_duration'] = 10

        self['hand_opening_duration'] = 2

        self['hand_max_aperture'] = 0.25

        self['isInPositionControl'] = False

        self['IMUGrasp'] = True

        ####################################################################################
        # IIT specific params for surface grasp
        ####################################################################################

        self['surface_grasp']['object']['hand_transform'] = tra.translation_matrix([0.0, 0.0, 0.15])

        self['surface_grasp']['object']['object_approach_transform'] = tra.translation_matrix([0.0, 0.0, 0.1])

        self['surface_grasp']['object']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([-0.001, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['cucumber']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.015, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['punnet']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.025, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['netbag']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.015, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['object']['kp'] = 6

        ####################################################################################
        # IIT specific params for wall grasp
        ####################################################################################        

        self['wall_grasp']['object']['kp'] = 6

        self['wall_grasp']['object']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.025, 0])

        self['wall_grasp']['mango']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.04, 0])
        
        self['wall_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.005, 0, -0.01]),
                                                                 tra.rotation_matrix(math.radians(-5.), [0, 1, 0]))
        