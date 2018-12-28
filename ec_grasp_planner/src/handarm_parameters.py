#!/usr/bin/env python

import math
import itertools
import numpy as np
from tf import transformations as tra

# python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp WallGrasp wall_chewinggum
# python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp EdgeGrasp --edgedistance -0.007 edge_chewinggum/
# python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp SurfaceGrasp test_folder

class BaseHandArm(dict):
    def __init__(self):
        self['mesh_file'] = "Unknown"
        self['mesh_file_scale'] = 1.

        # strategy types
        self['WallGrasp'] = {}        
        self['SurfaceGrasp'] = {}
        self['EdgeGrasp'] = {}

        # surface grasp parameters for different objects
        # 'object' is the default parameter set
        self['SurfaceGrasp']['object'] = {}

        # wall grasp parameters for different objects
        self['WallGrasp']['object'] = {}

        self['EdgeGrasp']['object'] = {}

        self['success_estimator_timeout'] = 10

    def checkValidity(self):
        # This function should always be called after the constructor of any class inherited from BaseHandArm
        # This convenience function allows to combine multiple sanity checks to ensure the handarm_parameters are as intended.
        self.assertNoCopyMissing()

    def assertNoCopyMissing(self):
        strategies = ['WallGrasp', 'EdgeGrasp', 'SurfaceGrasp']
        for s, s_other in itertools.product(strategies, repeat=2):
            for k in self[s]:
                for k_other in self[s_other]:
                    if not k_other == k and self[s][k] is self[s_other][k_other]:
                        # unitended reference copy of dictionary.
                        # This probably means that some previously defined parameters were overwritten.
                        raise AssertionError("You probably forgot to call copy(): {0} and {1} are equal for {2}".format(
                            k,k_other,s))

class WAM(BaseHandArm):
    def __init__(self):
        super(WAM, self).__init__()

class RBOHand2(WAM):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1
        self['drop_off_config'] = np.array([-0.57148, 0.816213, -0.365673, 1.53765, 0.30308, 0.128965, 1.02467])
# This map defines all grasp parameter such as poses and configurations for a specific robot system


# Define this map for your system if you want to port the planner
#Rbo hand 2 (P24 fingers and rotated palm) mounted on WAM.
class RBOHandP24WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHandP24WAM, self).__init__()

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        # transformation between object frame and hand palm frame above the object- should not be changed per object
        # please don't set x and y position, this should be done in pre_grasp_transform
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.3]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0]))).dot(
                                                                                tra.concatenate_matrices(
                                                                                    tra.translation_matrix([0.0, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(10.0), [0, 1, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.05

        # synergy type for soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_synergy'] = 1

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 5

        # first motion after grasp, in hand palm frame
        self['SurfaceGrasp']['object']['post_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, math.radians(-15.), 0])

        # duration of post grasp twist
        self['SurfaceGrasp']['object']['post_grasp_rotation_duration'] = 2.

        # speed of lifting the object
        self['SurfaceGrasp']['object']['up_speed'] = 0.05

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 8


        #####################################################################################
        # below are parameters for wall grasp with P24 fingers (standard RBO hand)
        #####################################################################################
        
        self['WallGrasprasp']['object']['initial_goal'] = np.array([0.439999, 0.624437, -0.218715, 1.71695, -0.735594, 0.197093, -0.920799])


        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['WallGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
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
        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.23, 0, -0.14]), #23 cm above object, 15 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        math.radians(15.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                    tra.rotation_matrix(                #this makes the fingers point downwards
                        math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.25

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.4 #sliding speed

        self['WallGrasp']['object']['hand_closing_duration'] = 1.0
        
        self['WallGrasp']['object']['hand_closing_synergy'] = 1

        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 2

        # speed of lifting the object
        self['WallGrasp']['object']['up_speed'] = 0.05

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8
        
    

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
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.3]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0]))).dot(
                                                                                tra.concatenate_matrices(
                                                                                    tra.translation_matrix([-0.08, 0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(20.0), [0, 1, 0])))

class RBOHandP24_pulpyWAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)


class KUKA(BaseHandArm):
    def __init__(self, **kwargs):
        super(KUKA, self).__init__()


class RBOHandO2KUKA(KUKA):
    def __init__(self, **kwargs):
        super(RBOHandO2KUKA, self).__init__()

        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"

        self['mesh_file_scale'] = 0.1

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]), tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))
        
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.3]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(-90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.05

        self['SurfaceGrasp']['object']['corrective_lift_duration'] = 1.5

        self['SurfaceGrasp']['object']['up_speed'] = 0.05

        # synergy type for soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_synergy'] = 1

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 5

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 8

        # duration of placing the object
        self['SurfaceGrasp']['object']['place_duration'] = 5


        #####################################################################################
        # below are parameters for wall grasp with P24 fingers (standard RBO hand)
        #####################################################################################
        
        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['WallGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
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
        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.23, 0, -0.14]), #23 cm above object, 15 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                    tra.rotation_matrix(                #this makes the fingers point downwards
                        math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.05

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.05

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.05 #sliding speed

        self['WallGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['WallGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['WallGrasp']['object']['hand_closing_duration'] = 1.0
        
        self['WallGrasp']['object']['hand_closing_synergy'] = 1

        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 2    

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8   

        # duration of placing the object
        self['WallGrasp']['object']['place_duration'] = 5 

        
             
        
# class PISAHandKUKA(KUKA):
#     def __init__(self, **kwargs):
#         super(PISAHandKUKA, self).__init__()

#         # Placement pose reachable for the PISA hand
#         self['pre_placement_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]), tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))
        
#         ####################################################################################
#         # Params that define the grasping controller
#         ####################################################################################

#         self['SimplePositionControl'] = True

#         self['ImpedanceControl'] = False

#         self['IMUGrasp'] = False

#         ####################################################################################
#         # PISAHand specific params irrespective of grasp type and/or object type
#         ####################################################################################

#         # Controller timeouts
#         self['hand_closing_duration'] = 1

#         self['hand_opening_duration'] = 2

#         self['compensation_duration'] = 10

#         self['lift_duration'] = 7

#         self['place_duration'] = 4

#         # Hand properties
#         self['hand_max_aperture'] = 0.25

#         # TRIK controller speeds
#         self['down_IFCO_speed'] = 0.03

#         self['up_IFCO_speed'] = 0.03

#         self['down_tote_speed'] = 0.05


#         ####################################################################################
#         # PISA Hand specific params for surface grasp
#         ####################################################################################

#         self['SurfaceGrasp']['object']['hand_transform'] = tra.translation_matrix([0.0, 0.0, 0.15])

#         self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([-0.001, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

#         self['SurfaceGrasp']['cucumber']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.015, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

#         self['SurfaceGrasp']['punnet']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.035, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

#         self['SurfaceGrasp']['netbag']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([0.015, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

#         self['SurfaceGrasp']['object']['downward_force'] = 4.

#         self['SurfaceGrasp']['object']['kp'] = 6

#         #real WRONG VALUES
#         # self['SurfaceGrasp']['cucumber']['success_rate'] = 1.0
#         # self['SurfaceGrasp']['punnet']['success_rate'] = 0.
#         # self['SurfaceGrasp']['netbag']['success_rate'] = 1.
#         # self['SurfaceGrasp']['mango']['success_rate'] = 1.
#         # self['SurfaceGrasp']['salad']['success_rate'] = 1.

#         #fake
#         self['SurfaceGrasp']['cucumber']['success_rate'] = 1.
#         self['SurfaceGrasp']['punnet']['success_rate'] = 0.8
#         self['SurfaceGrasp']['netbag']['success_rate'] = 0.9
#         self['SurfaceGrasp']['mango']['success_rate'] = 0.9
#         self['SurfaceGrasp']['salad']['success_rate'] = 1.

#         ####################################################################################
#         # IIT specific params for wall grasp
#         ####################################################################################        

#         self['WallGrasp']['object']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.025, -0.12])

#         self['WallGrasp']['netbag']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.05, -0.13])

#         self['WallGrasp']['mango']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.05, -0.12])

#         self['WallGrasp']['punnet']['pre_approach_transform'] = tra.translation_matrix([-0.20, -0.06, -0.14])

#         self['WallGrasp']['object']['downward_force'] = 2.

#         self['WallGrasp']['object']['slide_speed'] = 0.03

#         self['WallGrasp']['object']['wall_force'] = 3.5

#         self['WallGrasp']['mango']['wall_force'] = 4

#         self['WallGrasp']['salad']['wall_force'] = 4.5

#         self['WallGrasp']['punnet']['wall_force'] = 6

#         self['rotate_duration'] = 4
        
#         # self['WallGrasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.005, 0, -0.01]),
#         #                                                          tra.rotation_matrix(math.radians(-5.), [0, 1, 0]))
        
#         #self['WallGrasp']['object']['post_grasp_transform'] = np.array([0, 0, -0.01, 0, -0.09, 0])
#         self['WallGrasp']['object']['post_grasp_transform'] = np.array([0, 0, -0.01, 0, 0, 0])

#         self['WallGrasp']['object']['kp'] = 6

#         #real WRONG VALUES
#         # self['WallGrasp']['cucumber']['success_rate'] = 1.0
#         # self['WallGrasp']['punnet']['success_rate'] = 0.
#         # self['WallGrasp']['netbag']['success_rate'] = 1.
#         # self['WallGrasp']['mango']['success_rate'] = 1.
#         # self['WallGrasp']['salad']['success_rate'] = 1.

#         #fake
#         self['WallGrasp']['cucumber']['success_rate'] = 0.8
#         self['WallGrasp']['punnet']['success_rate'] = 0.6
#         self['WallGrasp']['netbag']['success_rate'] = 0.8
#         self['WallGrasp']['mango']['success_rate'] = 0.5
#         self['WallGrasp']['salad']['success_rate'] = 0.6
        

# class PISAGripperKUKA(KUKA):
#     def __init__(self, **kwargs):
#         super(PISAGripperKUKA, self).__init__()

#         # Placement pose reachable for the PISA gripper

#         self['pre_placement_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]), tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

#         ####################################################################################
#         # Params that define the grasping controller
#         ####################################################################################

#         self['SimplePositionControl'] = True

#         self['ImpedanceControl'] = False


#         ####################################################################################
#         # PISAGripper specific params irrespective of grasp type and/or object type
#         ####################################################################################

#         # Controller timeouts
#         self['hand_closing_duration'] = 1

#         self['hand_opening_duration'] = 2

#         self['lift_duration'] = 7

#         self['place_duration'] = 4

#         # Hand properties
#         self['hand_max_aperture'] = 0.25

#         # TRIK controller speeds
#         self['down_IFCO_speed'] = 0.03

#         self['up_IFCO_speed'] = 0.03

#         self['down_tote_speed'] = 0.05


#         ####################################################################################
#         # Gripper specific params for surface grasp
#         ####################################################################################

#         self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0, 0.0]),
#                                                                                 tra.concatenate_matrices(
#                                                                                     tra.rotation_matrix(
#                                                                                         math.radians(90.), [0, 0, 1]),
#                                                                                     tra.rotation_matrix(
#                                                                                         math.radians(180.), [1, 0, 0])))

#         self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, -0.12])

#         self['SurfaceGrasp']['punnet']['ee_in_goal_frame'] = tra.concatenate_matrices(tra.translation_matrix([0.01, -0.04, -0.12]), tra.rotation_matrix(
#                                                                                         math.radians(-15.), [1, 0, 0]) )

#         self['SurfaceGrasp']['object']['downward_force'] = 4

        
#         self['SurfaceGrasp']['object']['kp'] = 6

#         #real
#         # self['SurfaceGrasp']['cucumber']['success_rate'] = 1.0
#         # self['SurfaceGrasp']['punnet']['success_rate'] = 0.
#         # self['SurfaceGrasp']['netbag']['success_rate'] = 1.
#         # self['SurfaceGrasp']['mango']['success_rate'] = 1.
#         # self['SurfaceGrasp']['salad']['success_rate'] = 1.

#         #fake
#         self['SurfaceGrasp']['cucumber']['success_rate'] = 1.
#         self['SurfaceGrasp']['punnet']['success_rate'] = 0.
#         self['SurfaceGrasp']['netbag']['success_rate'] = 1.
#         self['SurfaceGrasp']['mango']['success_rate'] = 1.
#         self['SurfaceGrasp']['salad']['success_rate'] = 1.


#         ####################################################################################
#         # Gripper specific params for wall grasp
#         #################################################################################### 

#         scooping_angle_deg = 20        

#         self['WallGrasp']['object']['scooping_angle_deg'] = scooping_angle_deg

#         self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.07]),tra.rotation_matrix(
#                                                                                         math.radians(scooping_angle_deg), [0, 1, 0]))

#         self['WallGrasp']['netbag']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.03]),tra.rotation_matrix(
#                                                                                         math.radians(scooping_angle_deg), [0, 1, 0]))
        
#         self['WallGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.15, 0, -0.04]),tra.rotation_matrix(
#                                                                                         math.radians(scooping_angle_deg), [0, 1, 0]))
        
#         self['WallGrasp']['punnet']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0.03, -0.07]),tra.rotation_matrix(
#                                                                                         math.radians(scooping_angle_deg), [0, 1, 0]))
#         self['WallGrasp']['mango']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.04]),tra.rotation_matrix(
#                                                                                         math.radians(scooping_angle_deg), [0, 1, 0]))
        
#         self['WallGrasp']['salad']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.20, 0, -0.04]),tra.rotation_matrix(
#                                                                                         math.radians(scooping_angle_deg), [0, 1, 0]))
#         self['WallGrasp']['object']['downward_force'] = 2

#         self['short_lift_duration'] = 1.5

#         self['WallGrasp']['object']['slide_speed'] = 0.05

#         self['WallGrasp']['object']['wall_force'] = 5.5

#         self['WallGrasp']['cucumber']['wall_force'] = 12

#         self['WallGrasp']['mango']['wall_force'] = 12
        
#         self['WallGrasp']['punnet']['wall_force'] = 9

#         self['WallGrasp']['object']['kp'] = 6

#         #real
#         # self['WallGrasp']['cucumber']['success_rate'] = 1.0
#         # self['WallGrasp']['punnet']['success_rate'] = 0.
#         # self['WallGrasp']['netbag']['success_rate'] = 1.
#         # self['WallGrasp']['mango']['success_rate'] = 1.
#         # self['WallGrasp']['salad']['success_rate'] = 1.

#         #fake
#         self['WallGrasp']['cucumber']['success_rate'] = 0.8
#         self['WallGrasp']['punnet']['success_rate'] = 0.
#         self['WallGrasp']['netbag']['success_rate'] = 1.
#         self['WallGrasp']['mango']['success_rate'] = 1.
#         self['WallGrasp']['salad']['success_rate'] = 1.


# class ClashHandKUKA(KUKA):
#     def __init__(self, **kwargs):
#         super(ClashHandKUKA, self).__init__()

#         self['WallGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
#                 tra.rotation_matrix(
#                     math.radians(180.), [1, 0, 0]),
#                 tra.rotation_matrix(
#                     math.radians(90.0), [0, 0, 1]),
#                 tra.rotation_matrix(
#                     math.radians(0.0), [0, 1, 0])
#             )

#         ####################################################################################
#         # CLASH specific params irrespective of grasp type and/or object type
#         ####################################################################################

#         # Controller timeouts
#         self['hand_closing_duration'] = 2

#         self['hand_opening_duration'] = 2

#         self['lift_duration'] = 4

#         self['place_duration'] = 4

#         # TRIK controller speeds
#         self['down_IFCO_speed'] = 0.02

#         self['up_IFCO_speed'] = 0.03

#         self['down_tote_speed'] = 0.05


#         ####################################################################################
#         # CLASH specific params for surface grasp
#         ####################################################################################

#         self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0, 0.0]),
#                                                                                 tra.concatenate_matrices(                                                                                    
#                                                                                     tra.rotation_matrix(
#                                                                                         math.radians(180.), [1, 0, 0]),tra.rotation_matrix(
#                                                                                         math.radians(0.), [0, 0, 1])))

#         self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.005, -0.2])

#         self['SurfaceGrasp']['punnet']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.015, -0.2])

#         self['SurfaceGrasp']['netbag']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.015, -0.2])
        

#         self['SurfaceGrasp']['object']['downward_force'] = 2.5

#         self['SurfaceGrasp']['salad']['downward_force'] = 3

#         self['SurfaceGrasp']['salad']['thumb_pos_preshape'] = np.array([ 0, 0, 0])

#         self['SurfaceGrasp']['salad']['diff_pos_preshape'] = np.array([0, 0, 0])

#         self['SurfaceGrasp']['punnet']['thumb_pos_preshape'] = np.array([ 0, -30, 0])

#         self['SurfaceGrasp']['punnet']['diff_pos_preshape'] = np.array([-20, -20, 0])

#         self['SurfaceGrasp']['mango']['thumb_pos_preshape'] = np.array([ 0, 10, 0])

#         self['SurfaceGrasp']['mango']['diff_pos_preshape'] = np.array([5, 5, 5])

#         self['SurfaceGrasp']['cucumber']['thumb_pos_preshape'] = np.array([ 0, 10, 10])

#         self['SurfaceGrasp']['cucumber']['diff_pos_preshape'] = np.array([10, 10, 10])

#         self['SurfaceGrasp']['netbag']['thumb_pos_preshape'] = np.array([ 0, 10, 10])

#         self['SurfaceGrasp']['netbag']['diff_pos_preshape'] = np.array([10, 10, 10])



#         self['SurfaceGrasp']['mango']['short_lift_duration'] = 1.8

#         self['SurfaceGrasp']['mango']['thumb_pos'] = np.array([0, 30, 50])

#         self['SurfaceGrasp']['mango']['diff_pos'] = np.array([30, 30, 50])


#         self['SurfaceGrasp']['netbag']['short_lift_duration'] = 1.4

#         self['SurfaceGrasp']['netbag']['thumb_pos'] = np.array([0, 50, 50])

#         self['SurfaceGrasp']['netbag']['diff_pos'] = np.array([50, 50, 60])


#         self['SurfaceGrasp']['punnet']['short_lift_duration'] = 2.5

#         self['SurfaceGrasp']['punnet']['thumb_pos'] = np.array([0, 10, 50])

#         self['SurfaceGrasp']['punnet']['diff_pos'] = np.array([10, 10, 60])

#         self['SurfaceGrasp']['cucumber']['short_lift_duration'] = 1.4

#         self['SurfaceGrasp']['cucumber']['thumb_pos'] = np.array([0, 60, 30])

#         self['SurfaceGrasp']['cucumber']['diff_pos'] = np.array([60, 60, 30])

#         self['SurfaceGrasp']['salad']['short_lift_duration'] = 0

#         self['SurfaceGrasp']['salad']['thumb_pos'] = np.array([0, 60, 30])

#         self['SurfaceGrasp']['salad']['diff_pos'] = np.array([50, 50, 30])
        

#         ####################################################################################
#         # CLASH specific params for wall grasp
#         ####################################################################################        
        
#         scooping_angle_deg = 30

#         self['WallGrasp']['object']['scooping_angle_deg'] = scooping_angle_deg

#         # self['WallGrasp']['object']['scooping_angle_deg'] = 10

#         # self['WallGrasp']['mango']['scooping_angle_deg'] = 20

#         # self['WallGrasp']['salad']['scooping_angle_deg'] = 30  

#         self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.2, 0, -0.2]), tra.rotation_matrix(
#                                                                                         math.radians(scooping_angle_deg), [0, 1, 0]), tra.rotation_matrix(math.radians(90.), [0, 0, 1]), tra.rotation_matrix(math.radians(180.0), [0, 0, 1]))

#         self['WallGrasp']['object']['downward_force'] = 1.

#         self['WallGrasp']['object']['thumb_pos_preshape'] = np.array([ 0, -10, 0])
        
#         self['WallGrasp']['punnet']['thumb_pos_preshape'] = np.array([ 0, -25, 0])

#         self['WallGrasp']['object']['slide_speed'] = 0.03

#         self['WallGrasp']['mango']['wall_force'] = 5

#         self['WallGrasp']['cucumber']['wall_force'] = 10

#         self['WallGrasp']['netbag']['wall_force'] = 4

#         self['WallGrasp']['punnet']['wall_force'] = 12

#         self['WallGrasp']['salad']['wall_force'] = 1.5
        
#         self['WallGrasp']['object']['post_grasp_transform'] = np.array([0, 0, -0.01, 0, 0, 0])
        
#         self['rotate_duration'] = 3
        
        
#         