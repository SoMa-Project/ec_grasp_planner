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
                                                                                        math.radians(180.), [1, 0, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(10.0), [0, 1, 0]))

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
        # This is what should be changed per object if needed...
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
                                                                                        math.radians(180.), [1, 0, 0])))
        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([-0.08, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(20.0), [0, 1, 0]))

class RBOHandP24_pulpyWAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)


class KUKA(BaseHandArm):
    def __init__(self, **kwargs):
        super(KUKA, self).__init__()
        self['recovery_speed'] = 0.03
        self['recovery_duration'] = 5
        self['recovery_placement_force'] = 1.5
        self['view_joint_config'] = np.array([-0.7, 0.9, 0, -0.7, 0, 0.9, 0])
        # duration of placing the object
        self['place_duration'] = 5

        self['place_speed'] = 0.05


class RBOHandO2KUKA(KUKA):
    def __init__(self, **kwargs):
        super(RBOHandO2KUKA, self).__init__()

        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"

        self['mesh_file_scale'] = 0.1

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.29692, -0.57419, 0.16603]), tra.quaternion_matrix([0.6986, -0.68501, -0.11607, -0.171]))

        # This should be the same for all objects
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.3]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(-90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        # This should be the same for all objects
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.03

        self['SurfaceGrasp']['object']['corrective_lift_duration'] = 1.5

        self['SurfaceGrasp']['object']['up_speed'] = 0.03

        # time of soft hand preshape
        self['SurfaceGrasp']['object']['hand_preshaping_duration'] = 0

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 5

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 8


        #####################################################################################
        # below are parameters for wall grasp with O2 fingers (Ocado RBO hand)
        #####################################################################################
        
        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        # This should be the same for all objects
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
        # This is what should be changed per object if needed...
        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.23, 0, -0.14]), #23 cm above object, 14 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                    tra.rotation_matrix(                #this makes the fingers point downwards
                        math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.03

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.03

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.03 #sliding speed

        self['WallGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['WallGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['WallGrasp']['object']['hand_closing_duration'] = 5.0
        

        self['WallGrasp']['object']['hand_preshaping_duration'] = 0.0
        
        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 2    

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8

        # modify grasp parameters for cuucumber
        # TODO: This is just an example...
        self['WallGrasp']['cucumber'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0, -0.1]), #23 cm above object, 10 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(                #this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
        ))


        #####################################################################################
        # below are parameters for corner grasp with O2 fingers (Ocado RBO hand)
        #####################################################################################
        
        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        # This should be the same for all objects
        self['CornerGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
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
        # This is what should be changed per object if needed...
        self['CornerGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.23, 0, -0.14]), #23 cm above object, 14 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                    tra.rotation_matrix(                #this makes the fingers point downwards
                        math.radians(0.0), [0, 0, 1]),
            ))

        self['CornerGrasp']['object']['downward_force'] = 1.8

        self['CornerGrasp']['object']['down_speed'] = 0.03

        self['CornerGrasp']['object']['corrective_lift_duration'] = 1.5

        self['CornerGrasp']['object']['up_speed'] = 0.03

        self['CornerGrasp']['object']['wall_force'] = 12.0

        self['CornerGrasp']['object']['slide_speed'] = 0.03 #sliding speed

        self['CornerGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['CornerGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['CornerGrasp']['object']['hand_closing_duration'] = 5.0
        

        self['CornerGrasp']['object']['hand_preshaping_duration'] = 0.0
        
        # first motion after grasp, in hand palm frame
        self['CornerGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['CornerGrasp']['object']['post_grasp_rotation_duration'] = 2    

        # duration of lifting the object
        self['CornerGrasp']['object']['lift_duration'] = 8

        # modify grasp parameters for cuucumber
        # TODO: This is just an example...
        self['CornerGrasp']['cucumber'] = self['CornerGrasp']['object'].copy()
        self['CornerGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0, -0.1]), #23 cm above object, 10 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(                #this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
        ))
        
            
        
class PISAHandKUKA(KUKA):
    def __init__(self, **kwargs):
        super(PISAHandKUKA, self).__init__()

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.40392, -0.65228, 0.21258]), tra.quaternion_matrix([-0.6846, 0.72715, 0.018816, 0.047258]))
        

        ####################################################################################
        # Params that define the grasping controller
        ####################################################################################

        self['SimplePositionControl'] = True

        self['ImpedanceControl'] = False
        self['hand_max_aperture'] = 0.15
        self['SurfaceGrasp']['object']['kp'] = 0.03

        self['IMUGrasp'] = False
        self['compensation_duration'] = 2

        ####################################################################################
        # PISAHand specific params
        ####################################################################################

        self['SurfaceGrasp']['object']['hand_preshape_goal'] = 0

        self['SurfaceGrasp']['object']['hand_preshaping_duration'] = 0

        # This should be the same for all objects
        self['SurfaceGrasp']['object']['hand_transform'] = tra.translation_matrix([0.0, 0.0, 0.2])

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        # This should be the same for all objects
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.inverse_matrix(tra.translation_matrix([-0.001, -0.002, 0.003]).dot(tra.quaternion_matrix([0.595, 0.803, -0.024, -0.013])))

        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.03

        self['SurfaceGrasp']['object']['up_speed'] = 0.03

        # synergy type for soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_goal'] = 0.9

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 1

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 11

        self['SurfaceGrasp']['cucumber'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['cucumber']['hand_preshape_goal'] = 0.55
        self['SurfaceGrasp']['cucumber']['hand_closing_goal'] = 1

        #####################################################################################
        # below are parameters for wall grasp 
        #####################################################################################
        self['WallGrasp']['object']['hand_preshape_goal'] = 0
        self['WallGrasp']['object']['hand_preshaping_duration'] = 0

        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        # This should be the same for all objects
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
        # This should be changed per object if needed
        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.15, 0, -0.14]), #15 cm above object, 15 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                    tra.rotation_matrix(                #this makes the fingers point downwards
                        math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.03

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.03

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.03 #sliding speed

        self['WallGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['WallGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['WallGrasp']['object']['hand_closing_duration'] = 1.0
        
        self['WallGrasp']['object']['hand_closing_goal'] = 0.8

        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 2    

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8   
        

class PISAGripperKUKA(KUKA):
    def __init__(self, **kwargs):
        super(PISAGripperKUKA, self).__init__()

        # Placement pose reachable for the PISA gripper

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.32804, -0.62733, 0.068286]), tra.quaternion_matrix([0.85531, -0.51811, -0.0023802, -0.0016251]))

        ####################################################################################
        # Params that define the grasping controller
        ####################################################################################

        self['SimplePositionControl'] = True

        self['ImpedanceControl'] = False
        self['hand_max_aperture'] = 0.15
        self['SurfaceGrasp']['object']['kp'] = 0.03


        ####################################################################################
        # PISAGripper specific params
        ####################################################################################

        self['SurfaceGrasp']['object']['hand_preshape_goal'] = 0
        self['SurfaceGrasp']['object']['hand_preshaping_duration'] = 0

        # This should be the same for all objects
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.15]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(-90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        # This should be the same for all objects
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.03

        self['SurfaceGrasp']['object']['up_speed'] = 0.03

        # synergy type for soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_goal'] = 0.6

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 1

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 11

        self['SurfaceGrasp']['cucumber'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['cucumber']['hand_preshape_goal'] = 0.15
        self['SurfaceGrasp']['cucumber']['downward_force'] = 10



        #####################################################################################
        # below are parameters for wall grasp 
        #####################################################################################
        self['WallGrasp']['object']['hand_preshape_goal'] = 0
        self['WallGrasp']['object']['hand_preshaping_duration'] = 0

        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        # This should be the same for all objects
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
        # This should be changed per object if needed...

        self['WallGrasp']['object']['scooping_angle'] = math.radians(20)

        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.15, 0, -0.1]), #15 cm above object, 10 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        self['WallGrasp']['object']['scooping_angle'], [0, 1, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.03

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.03

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.03 #sliding speed

        self['WallGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['WallGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['WallGrasp']['object']['hand_closing_duration'] = 1.0
        
        self['WallGrasp']['object']['hand_closing_goal'] = 0.8

        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 2    

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8  

class ClashHandKUKA(KUKA):
    def __init__(self, **kwargs):
        super(ClashHandKUKA, self).__init__()

        self['view_joint_config'] = np.array([-0.6, 0.9, -0.3, -0.6, 1.5, 0.9, 1.5])

        # Placement pose reachable for the CLASH hand

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.32804, -0.62733, 0.068286]), tra.quaternion_matrix([0.85531, -0.51811, -0.0023802, -0.0016251]))

        ####################################################################################
        # CLASH hand specific params
        ####################################################################################

        # This should be the same for all objects
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.15]),
                                                                                tra.concatenate_matrices(
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(-90.), [0, 0, 1]),
                                                                                    tra.rotation_matrix(
                                                                                        math.radians(180.), [1, 0, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        # This should be the same for all objects
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.0, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 3

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.03

        self['SurfaceGrasp']['object']['up_speed'] = 0.03

        self['SurfaceGrasp']['object']['corrective_lift_duration'] = 1.5

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 2

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 8

        # default hand close value
        self['SurfaceGrasp']['object']['goal_close'] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # default hand preshape value
        self['SurfaceGrasp']['object']['goal_preshape'] = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

        self['SurfaceGrasp']['object']['thumb_stiffness'] = np.array([0.25])
        self['SurfaceGrasp']['object']['diff_stiffness'] = np.array([0.25])

        self['SurfaceGrasp']['mango'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['mango']['goal_preshape'] = np.array([ 0, 10, 0, 5, 5, 5])
        self['SurfaceGrasp']['mango']['goal_close'] = np.array([0, 30, 60, 50, 50, 60])

        # self['SurfaceGrasp']['cucumber'] = self['SurfaceGrasp']['object'].copy() #params used for the experiment
        # self['SurfaceGrasp']['cucumber']['goal_preshape'] = np.array([ 0, 10, 5, 10, 10, 0])
        # self['SurfaceGrasp']['cucumber']['goal_close'] = np.array([0, 60, 30, 60, 60, 30])
        # self['SurfaceGrasp']['cucumber']['corrective_lift_duration'] = 1

        self['SurfaceGrasp']['cucumber'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['cucumber']['goal_preshape'] = np.array([ 0, 30, 10, 50, 50, 10])        
        self['SurfaceGrasp']['cucumber']['goal_close'] = np.array([0, 80, 30, 80, 80, 30])
        self['SurfaceGrasp']['cucumber']['corrective_lift_duration'] = 1.2
        self['SurfaceGrasp']['cucumber']['downward_force'] = 4

        self['SurfaceGrasp']['punnet'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['punnet']['goal_preshape'] = np.array([ 0, -20, 0, -30, -30, -10])
        self['SurfaceGrasp']['punnet']['goal_close'] = np.array([0, 20, 50, 30, 30, 60])
        self['SurfaceGrasp']['punnet']['pre_approach_transform'] = tra.concatenate_matrices(tra.translation_matrix([0.02, 0.0, 0.0]),
                                                                                    tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))
        self['SurfaceGrasp']['punnet']['corrective_lift_duration'] = 1.2

        self['SurfaceGrasp']['lettuce'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['lettuce']['goal_preshape'] = np.array([ 0, -20, 0, -30, -30, -10])
        self['SurfaceGrasp']['lettuce']['goal_close'] = np.array([0, 60, 50, 60, 60, 50])
        self['SurfaceGrasp']['lettuce']['corrective_lift_duration'] = 0

        # self['SurfaceGrasp']['netbag'] = self['SurfaceGrasp']['object'].copy()
        # self['SurfaceGrasp']['netbag']['goal_preshape'] = np.array([ 0, 10, 10, 10, 10, 10])
        # self['SurfaceGrasp']['netbag']['goal_close'] = np.array([0, 50, 70, 60, 60, 60])
        # self['SurfaceGrasp']['netbag']['corrective_lift_duration'] = 1.2

        self['SurfaceGrasp']['netbag'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['netbag']['goal_preshape'] = np.array([ 0, 30, 0, 30, 30, 10])
        self['SurfaceGrasp']['netbag']['goal_close'] = np.array([0, 60, 90, 70, 70, 90])
        self['SurfaceGrasp']['netbag']['corrective_lift_duration'] = 1.2
        self['SurfaceGrasp']['netbag']['downward_force'] = 4

        #####################################################################################
        # below are parameters for wall grasp 
        #####################################################################################
       
        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        # This should be the same for all objects
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
        # This should be changed per object if needed...

        self['WallGrasp']['object']['scooping_angle'] = math.radians(30)

        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.15, 0, -0.05]), #15 cm above object, 10 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        self['WallGrasp']['object']['scooping_angle'], [0, 1, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.5

        self['WallGrasp']['object']['down_speed'] = 0.03

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.03

        self['WallGrasp']['object']['wall_force'] = 5.0

        self['WallGrasp']['object']['slide_speed'] = 0.03 #sliding speed

        self['WallGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['WallGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['WallGrasp']['object']['hand_closing_duration'] = 2
        
        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([0.0, 0.0, -0.01, 0.0, math.radians(0.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 3   

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8  

        # default hand close value
        self['WallGrasp']['object']['goal_close'] = np.array([0, 50, 30, 55, 50, 20])

        # default hand preshape value
        self['WallGrasp']['object']['goal_preshape'] = np.array([0, -10, 0, 10, 15, 0])

        # Object dependent parameters example
        self['WallGrasp']['object']['thumb_stiffness'] = np.array([0.25])
        self['WallGrasp']['object']['diff_stiffness'] = np.array([0.25])

        self['WallGrasp']['mango'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['mango']['scooping_angle'] = math.radians(10)
        self['WallGrasp']['mango']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.15, 0, -0.04]), #15 cm above object, 2 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        self['WallGrasp']['mango']['scooping_angle'], [0, 1, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 0, 1]),
            ))
        self['WallGrasp']['mango']['wall_force'] = 10.0
        self['WallGrasp']['mango']['goal_preshape'] = np.array([0, 0, 0, 40, 40, 10])
        self['WallGrasp']['mango']['goal_close'] = np.array([0, 65, 20, 60, 60, 30])
        self['WallGrasp']['mango']['lift_duration'] = 8



        self['WallGrasp']['cucumber'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['cucumber']['scooping_angle'] = math.radians(10)
        self['WallGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
                # tra.translation_matrix([-0.15, 0, -0.03]), #15 cm above object, 2 cm behind
                tra.translation_matrix([-0.15, 0, -0.03]),
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        self['WallGrasp']['cucumber']['scooping_angle'], [0, 1, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 0, 1]),
            ))
        self['WallGrasp']['cucumber']['wall_force'] = 13.0
        self['WallGrasp']['cucumber']['goal_preshape'] = np.array([0, 20, 0, 15, 15, 20])
        self['WallGrasp']['cucumber']['corrective_lift_duration'] = 1.0
        self['WallGrasp']['cucumber']['post_grasp_rotation_duration'] = 4
        self['WallGrasp']['cucumber']['lift_duration'] = 11


        self['WallGrasp']['punnet'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['punnet']['scooping_angle'] = math.radians(10)        
        self['WallGrasp']['punnet']['wall_force'] = 15
        self['WallGrasp']['punnet']['goal_preshape'] = np.array([0, 10, 0, 30, 30, -20])
        self['WallGrasp']['punnet']['corrective_lift_duration'] = 1.3
        self['WallGrasp']['punnet']['post_grasp_rotation_duration'] = 4
        self['WallGrasp']['punnet']['pre_approach_transform'] = tra.concatenate_matrices(
                # tra.translation_matrix([-0.15, 0, -0.03]), #15 cm above object, 2 cm behind
                tra.translation_matrix([-0.15, 0, -0.05]),
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        self['WallGrasp']['punnet']['scooping_angle'], [0, 1, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 0, 1]),
            ))
        self['WallGrasp']['punnet']['post_grasp_twist'] = np.array([0.0, 0.0, -0.01, 0.0, math.radians(0.0), 0.0])
        self['WallGrasp']['punnet']['slide_speed'] = 0.05 #sliding speed
        self['WallGrasp']['punnet']['diff_stiffness'] = np.array([0.0])


        self['WallGrasp']['lettuce'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['lettuce']['scooping_angle'] = math.radians(15)
        self['WallGrasp']['lettuce']['goal_preshape'] = np.array([0, 10, 0, 30, 30, -20])
        self['WallGrasp']['lettuce']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.15, 0, -0.07]), #15 cm above object, 2 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        self['WallGrasp']['lettuce']['scooping_angle'], [0, 1, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 0, 1]),
            ))
        self['WallGrasp']['lettuce']['wall_force'] = 8
        self['WallGrasp']['lettuce']['lift_duration'] = 12
        self['WallGrasp']['lettuce']['goal_close'] = np.array([0, 70, 20, 70, 70, 25])
        self['WallGrasp']['lettuce']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['netbag'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['netbag']['scooping_angle'] = math.radians(10)
        self['WallGrasp']['netbag']['goal_preshape'] = np.array([ 0, 10, 10, 25, 25, 0])
        self['WallGrasp']['netbag']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.15, 0, -0.07]), #15 cm above object, 2 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        self['WallGrasp']['netbag']['scooping_angle'], [0, 1, 0]),
                    tra.rotation_matrix(
                        math.radians(0.0), [0, 0, 1]),
            ))
        self['WallGrasp']['netbag']['wall_force'] = 7
        self['WallGrasp']['netbag']['lift_duration'] = 12
        self['WallGrasp']['netbag']['goal_close'] = np.array([0, 50, 70, 60, 60, 60])
        self['WallGrasp']['netbag']['corrective_lift_duration'] = 1.5
        self['WallGrasp']['netbag']['diff_stiffness'] = np.array([0.0])
        self['WallGrasp']['netbag']['downward_force'] = 2
