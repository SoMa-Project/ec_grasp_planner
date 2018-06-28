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
        self['surface_grasp'] = {}

        # surface grasp parameters for different objects
        # 'object' is the default parameter set
        self['surface_grasp']['object'] = {}
        self['surface_grasp']['cucumber'] = {}
        self['surface_grasp']['punnet'] = {}
        self['surface_grasp']['netbag'] = {}
        self['surface_grasp']['mango'] = {}
        self['surface_grasp']['salad'] = {}

        # wall grasp parameters for different objects
        self['wall_grasp']['object'] = {}
        self['wall_grasp']['punnet'] = {}
        self['wall_grasp']['salad'] = {}
        self['wall_grasp']['netbag'] = {}
        self['wall_grasp']['mango'] = {}
        self['wall_grasp']['cucumber'] = {}

        # object parameters uncorrelated to grasp type
        self['object'] = {}
        self['punnet'] = {}
        self['netbag'] = {}
        self['salad'] = {}
        self['cucumber'] = {}
        self['mango'] = {}


class KUKA(BaseHandArm):
    def __init__(self, **kwargs):
        super(KUKA, self).__init__()

        ####################################################################################
        # General params 
        ####################################################################################


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


class RBOHandO2KUKA(KUKA):
    def __init__(self, **kwargs):
        super(RBOHandO2KUKA, self).__init__()

        # Placement pose reachable for the RBO hand
        self['pre_placement_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]), tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

        ####################################################################################
        # RBOHand specific params irrespective of grasp type and/or object type
        ####################################################################################

        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"

        self['mesh_file_scale'] = 0.1
        
        self['hand_closing_duration'] = 6

        self['hand_opening_duration'] = 4

        self['lift_duration'] = 13

        self['place_duration'] = 5

        # TRIK controller speeds
        self['down_IFCO_speed'] = 0.03

        self['up_IFCO_speed'] = 0.03

        self['down_tote_speed'] = 0.05



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

        self['surface_grasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([-0.02, 0.02, 0.0])

        self['surface_grasp']['object']['downward_force'] = 2

        self['surface_grasp']['netbag']['downward_force'] = 1.5

        self['surface_grasp']['object']['short_lift_duration'] = 2


        ####################################################################################
        # RBO specific params for wall grasp
        ####################################################################################
        
        self['wall_grasp']['object']['pre_approach_transform'] = tra.translation_matrix([-0.20, 0, -0.12])

        self['wall_grasp']['netbag']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.01, -0.12])

        self['wall_grasp']['mango']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.01, -0.12])

        self['wall_grasp']['salad']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.01, -0.15])

        self['wall_grasp']['cucumber']['pre_approach_transform'] = tra.translation_matrix([-0.20, 0, -0.12])

        self['wall_grasp']['punnet']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.0, -0.14])

        self['wall_grasp']['object']['downward_force'] = 2.

        self['wall_grasp']['object']['short_lift_duration'] = 2

        self['wall_grasp']['object']['slide_speed'] = 0.02

        self['wall_grasp']['object']['wall_force'] = 2.5        
        
        self['wall_grasp']['cucumber']['wall_force'] = 3.5

        self['wall_grasp']['object']['short_slide_duration'] = 2     
             
        
class PISAHandKUKA(KUKA):
    def __init__(self, **kwargs):
        super(PISAHandKUKA, self).__init__()

        # Placement pose reachable for the PISA hand
        self['pre_placement_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]), tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

        ####################################################################################
        # Params that define the grasping controller
        ####################################################################################

        self['SimplePositionControl'] = True

        self['ImpedanceControl'] = False

        self['IMUGrasp'] = False

        ####################################################################################
        # PISAHand specific params irrespective of grasp type and/or object type
        ####################################################################################

        # Controller timeouts
        self['hand_closing_duration'] = 2

        self['hand_opening_duration'] = 2

        self['compensation_duration'] = 10

        self['lift_duration'] = 13

        self['place_duration'] = 5

        # Hand properties
        self['hand_max_aperture'] = 0.25

        # TRIK controller speeds
        self['down_IFCO_speed'] = 0.03

        self['up_IFCO_speed'] = 0.03

        self['down_tote_speed'] = 0.05


        ####################################################################################
        # PISA Hand specific params for surface grasp
        ####################################################################################

        self['surface_grasp']['object']['hand_transform'] = tra.translation_matrix([0.0, 0.0, 0.15])

        self['surface_grasp']['object']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([-0.001, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['cucumber']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.015, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['punnet']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.025, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['netbag']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.015, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        self['surface_grasp']['object']['downward_force'] = 4.

        self['surface_grasp']['object']['kp'] = 6

        ####################################################################################
        # IIT specific params for wall grasp
        ####################################################################################        

        self['wall_grasp']['object']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.025, 0])

        self['wall_grasp']['mango']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.04, 0])

        self['wall_grasp']['object']['downward_force'] = 2.

        self['wall_grasp']['object']['slide_speed'] = 0.05

        self['wall_grasp']['object']['wall_force'] = 3.5

        self['rotate_duration'] = 3
        
        self['wall_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.005, 0, -0.01]),
                                                                 tra.rotation_matrix(math.radians(-5.), [0, 1, 0]))
        
        self['wall_grasp']['object']['kp'] = 6
        

class PISAGripperKUKA(KUKA):
    def __init__(self, **kwargs):
        super(PISAGripperKUKA, self).__init__()

        # Placement pose reachable for the PISA gripper

        self['pre_placement_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]), tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

        ####################################################################################
        # Params that define the grasping controller
        ####################################################################################

        self['SimplePositionControl'] = True

        self['ImpedanceControl'] = False


        ####################################################################################
        # PISAGripper specific params irrespective of grasp type and/or object type
        ####################################################################################

        # Controller timeouts
        self['hand_closing_duration'] = 2

        self['hand_opening_duration'] = 2

        self['lift_duration'] = 13

        self['place_duration'] = 5

        # Hand properties
        self['hand_max_aperture'] = 0.25

        # TRIK controller speeds
        self['down_IFCO_speed'] = 0.03

        self['up_IFCO_speed'] = 0.03

        self['down_tote_speed'] = 0.05


        ####################################################################################
        # Gripper specific params for surface grasp
        ####################################################################################

        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0, 0.0]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        self['surface_grasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, -0.12])

        self['surface_grasp']['punnet']['ee_in_goal_frame'] = tra.concatenate_matrices(tra.translation_matrix([0.01, -0.04, -0.12]), tra.rotation_matrix(
                                                                                        math.radians(-15.), [1, 0, 0]) )

        self['surface_grasp']['object']['downward_force'] = 4

        
        self['surface_grasp']['object']['kp'] = 6

        ####################################################################################
        # Gripper specific params for wall grasp
        #################################################################################### 

        scooping_angle_deg = 20        

        self['wall_grasp']['object']['scooping_angle_deg'] = scooping_angle_deg

        self['wall_grasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.07]),tra.rotation_matrix(
                                                                                        math.radians(scooping_angle_deg), [0, 1, 0]))

        self['wall_grasp']['netbag']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.03]),tra.rotation_matrix(
                                                                                        math.radians(scooping_angle_deg), [0, 1, 0]))
        
        self['wall_grasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.15, 0, -0.01]),tra.rotation_matrix(
                                                                                        math.radians(scooping_angle_deg), [0, 1, 0]))
        
        self['wall_grasp']['punnet']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0.03, -0.07]),tra.rotation_matrix(
                                                                                        math.radians(scooping_angle_deg), [0, 1, 0]))
        self['wall_grasp']['mango']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.04]),tra.rotation_matrix(
                                                                                        math.radians(scooping_angle_deg), [0, 1, 0]))
        
        self['wall_grasp']['salad']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.04]),tra.rotation_matrix(
                                                                                        math.radians(scooping_angle_deg), [0, 1, 0]))
        self['wall_grasp']['object']['downward_force'] = 2

        self['short_lift_duration'] = 1.5

        self['wall_grasp']['object']['slide_speed'] = 0.05

        self['wall_grasp']['object']['wall_force'] = 5.5

        self['wall_grasp']['cucumber']['wall_force'] = 9

        self['wall_grasp']['mango']['wall_force'] = 12
        
        self['wall_grasp']['punnet']['wall_force'] = 9

        self['wall_grasp']['object']['kp'] = 6

class ClashHandKUKA(KUKA):
    def __init__(self, **kwargs):
        super(ClashHandKUKA, self).__init__()

        # Placement pose reachable for the CLASH hand
        self['pre_placement_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.562259, 0.468926, 0.299963]), tra.quaternion_matrix([0.977061, -0.20029, 0.05786, 0.04345]))

        ####################################################################################
        # CLASH specific params irrespective of grasp type and/or object type
        ####################################################################################

        # Controller timeouts
        self['hand_closing_duration'] = 2

        self['hand_opening_duration'] = 2

        self['lift_duration'] = 10

        self['place_duration'] = 3

        # TRIK controller speeds
        self['down_IFCO_speed'] = 0.02

        self['up_IFCO_speed'] = 0.03

        self['down_tote_speed'] = 0.05


        ####################################################################################
        # CLASH specific params for surface grasp
        ####################################################################################

        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0, 0.0]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(0.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        self['surface_grasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.005, -0.15])

        self['surface_grasp']['punnet']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.015, -0.15])

        self['surface_grasp']['netbag']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.015, -0.15])
        

        self['surface_grasp']['object']['downward_force'] = 2.5

        self['surface_grasp']['salad']['downward_force'] = 3

        self['surface_grasp']['salad']['thumb_pos_preshape'] = np.array([ 0, 0, 0])

        self['surface_grasp']['salad']['diff_pos_preshape'] = np.array([0, 0, 0])

        self['surface_grasp']['punnet']['thumb_pos_preshape'] = np.array([ 0, -30, 0])

        self['surface_grasp']['punnet']['diff_pos_preshape'] = np.array([-20, -20, 0])

        self['surface_grasp']['mango']['thumb_pos_preshape'] = np.array([ 0, 10, 0])

        self['surface_grasp']['mango']['diff_pos_preshape'] = np.array([5, 5, 5])

        self['surface_grasp']['cucumber']['thumb_pos_preshape'] = np.array([ 0, 10, 10])

        self['surface_grasp']['cucumber']['diff_pos_preshape'] = np.array([10, 10, 10])

        self['surface_grasp']['netbag']['thumb_pos_preshape'] = np.array([ 0, 10, 10])

        self['surface_grasp']['netbag']['diff_pos_preshape'] = np.array([10, 10, 10])



        self['surface_grasp']['mango']['short_lift_duration'] = 1.8

        self['surface_grasp']['mango']['thumb_pos'] = np.array([0, 30, 50])

        self['surface_grasp']['mango']['diff_pos'] = np.array([30, 30, 50])


        self['surface_grasp']['netbag']['short_lift_duration'] = 1.4

        self['surface_grasp']['netbag']['thumb_pos'] = np.array([0, 50, 50])

        self['surface_grasp']['netbag']['diff_pos'] = np.array([50, 50, 60])


        self['surface_grasp']['punnet']['short_lift_duration'] = 2.5

        self['surface_grasp']['punnet']['thumb_pos'] = np.array([0, 10, 50])

        self['surface_grasp']['punnet']['diff_pos'] = np.array([10, 10, 60])

        self['surface_grasp']['cucumber']['short_lift_duration'] = 1.4

        self['surface_grasp']['cucumber']['thumb_pos'] = np.array([0, 60, 30])

        self['surface_grasp']['cucumber']['diff_pos'] = np.array([60, 60, 30])

        self['surface_grasp']['salad']['short_lift_duration'] = 0

        self['surface_grasp']['salad']['thumb_pos'] = np.array([0, 60, 30])

        self['surface_grasp']['salad']['diff_pos'] = np.array([50, 50, 30])
        

        ####################################################################################
        # CLASH specific params for wall grasp
        ####################################################################################        

        self['wall_grasp']['object']['scooping_angle_deg'] = 10

        self['wall_grasp']['mango']['scooping_angle_deg'] = 20

        self['wall_grasp']['salad']['scooping_angle_deg'] = 30  

        self['wall_grasp']['object']['downward_force'] = 1.5

        self['wall_grasp']['object']['thumb_pos_preshape'] = np.array([ 0, -10, 0])
        
        self['wall_grasp']['punnet']['thumb_pos_preshape'] = np.array([ 0, -25, 0])

        self['wall_grasp']['object']['slide_speed'] = 0.03

        self['wall_grasp']['mango']['wall_force'] = 5

        self['wall_grasp']['cucumber']['wall_force'] = 10

        self['wall_grasp']['netbag']['wall_force'] = 4

        self['wall_grasp']['punnet']['wall_force'] = 12

        self['wall_grasp']['salad']['wall_force'] = 1.5
        
        self['wall_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.005, 0, -0.01]),
                                                                 tra.rotation_matrix(math.radians(0.), [0, 1, 0]))
        
        self['rotate_duration'] = 3
        
        
        