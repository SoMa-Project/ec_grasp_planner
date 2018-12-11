#!/usr/bin/env python

import math
import itertools
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

        # surface grasp parameters for different objects
        # 'object' is the default parameter set
        self['surface_grasp']['object'] = {}

        # wall grasp parameters for differnt objects
        self['wall_grasp']['object'] = {}

        # wall grasp parameters for differnt objects
        self['edge_grasp']['object'] = {}

        self['isForceControllerAvailable'] = False

    def checkValidity(self):
        # This function should always be called after the constructor of any class inherited from BaseHandArm
        # This convenience function allows to combine multiple sanity checks to ensure the handarm_parameters are as intended.
        self.assertNoCopyMissing()

    def assertNoCopyMissing(self):
        strategies = ['wall_grasp', 'edge_grasp', 'surface_grasp']
        for s, s_other in itertools.product(strategies, repeat=2):
            for k in self[s]:
                for k_other in self[s_other]:
                    if not k_other == k and self[s][k] is self[s_other][k_other]:
                        # unitended reference copy of dictionary.
                        # This probably means that some previously defined parameters were overwritten.
                        raise AssertionError("You probably forgot to call copy(): {0} and {1} are equal for {2}".format(
                            k, k_other, s))


class RBOHand2(BaseHandArm):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1


# This map defines all grasp parameter such as poses and configurations for a specific robot system


# Define this map for your system if you want to port the planner
# Rbo hand 2 (P24 fingers and rotated palm) mounted on WAM.
class RBOHandP24WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHandP24WAM, self).__init__()

        # does the robot support impedance control
        self['isForceControllerAvailable'] = True

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        # transformation between object frame and hand palm frame above the object- should not be changed per object
        # please don't set x and y position, this should be done in pre_grasp_transform
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.3]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(90.), [0, 0, 1]),
                tra.rotation_matrix(
                    math.radians(180.), [1, 0, 0])))

        # position of hand relative to the object before and at grasping
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]), tra.rotation_matrix(math.radians(10.0), [0, 1, 0]))

        # first motion after grasp, in hand palm frame
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(-15.),
                                [0, 1, 0]))

        # second motion after grasp, in hand palm frame
        self['surface_grasp']['object']['go_up_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.03, 0, -0.3]),
            tra.rotation_matrix(math.radians(-20.),
                                [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['surface_grasp']['object']['downward_force'] = 4

        # drop configuration - this is system specific!
        self['surface_grasp']['object']['drop_off_config'] = np.array(
            [-0.57148, 0.816213, -0.365673, 1.53765, 0.30308, 0.128965, 1.02467])

        # synergy type for soft hand closing
        self['surface_grasp']['object']['hand_closing_synergy'] = 1

        # time of soft hand closing
        self['surface_grasp']['object']['hand_closing_duration'] = 5

        # time of soft hand closing
        self['surface_grasp']['object']['down_speed'] = 0.35
        self['surface_grasp']['object']['up_speed'] = 0.409
        self['surface_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.09])  # first value: rotational, second translational
        self['surface_grasp']['object']['pre_grasp_velocity'] = np.array([0.125, 0.08])

        # defines the manifold in which alternative goal poses are sampled during kinematic checks
        self['surface_grasp']['object']['pre_grasp_manifold'] = {'min_position_deltas': [-0.01, -0.01, -0.01]}
        # TODO change this to a class to ensure every item is set? also make sure the copies are deep copies and the sanity check is done recursively...!
        self['surface_grasp']['object']['max_position_deltas'] = [0.01, 0.01, 0.01]
        self['surface_grasp']['object']['min_orientation_deltas'] = [0, 0, -1.5]
        self['surface_grasp']['object']['max_orientation_deltas'] = [0, 0, 1.5]

        #####################################################################################
        # below are parameters for wall grasp with P24 fingers (standard RBO hand)
        #####################################################################################
        self['wall_grasp']['object']['hand_closing_duration'] = 1.0
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
            tra.translation_matrix([-0.23, 0, -0.14]),  # 23 cm above object, 15 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(15.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        # first motion after grasp, in hand palm frame
        self['wall_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.05, 0.0, 0.0]),  # nothing right now
            tra.rotation_matrix(math.radians(-18.0),
                                [0, 1, 0]))

        # drop configuration - this is system specific!
        self['wall_grasp']['object']['drop_off_config'] = np.array(
            [-0.318009, 0.980688, -0.538178, 1.67298, -2.07823, 0.515781, -0.515471])

        self['wall_grasp']['object']['table_force'] = 1.8
        self['wall_grasp']['object']['lift_dist'] = 0.1  # short lift after initial contact (before slide)
        self['wall_grasp']['object']['sliding_dist'] = 0.4  # sliding distance, should be min. half Ifco size
        self['wall_grasp']['object']['up_dist'] = 0.25
        self['wall_grasp']['object']['down_dist'] = 0.25
        self['wall_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.09])  # first value: rotational, second translational
        self['wall_grasp']['object']['slide_velocity'] = np.array([0.125, 0.30])  # np.array([0.125, 0.12])
        self['wall_grasp']['object']['wall_force'] = 12.0


class RBOHandP11WAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)


class RBOHandP24_opposableThumbWAM(RBOHand2):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)


# Rbo hand 2 (Ocado version with long fingers and rotated palm) mounted on WAM.
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


# TODO: this needs to be adapted similar to match the frames above!
# The map is now 3d and the frame definitions changed.
class RBOHand2Kuka(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2Kuka, self).__init__()

        # old parameters below - must be updated to new convention!
        self['surface_grasp']['initial_goal'] = np.array(
            [-0.05864322834179703, 0.4118988657714642, -0.05864200146127985, -1.6887810963180838, -0.11728653060066829,
             -0.8237944986945402, 0])
        self['surface_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.2])
        self['surface_grasp']['hand_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0]),
                                                                      tra.rotation_matrix(math.radians(0.), [0, 0, 1]))
        self['surface_grasp']['downward_force'] = 7.
        self['surface_grasp']['valve_pattern'] = (
        np.array([[0., 4.1], [0., 0.1], [0., 5.], [0., 5.], [0., 2.], [0., 3.5]]), np.array([[1, 0]] * 6))

        self['wall_grasp']['pregrasp_pose'] = tra.translation_matrix([0.05, 0, -0.2])
        self['wall_grasp']['table_force'] = 7.0
        self['wall_grasp']['sliding_speed'] = 0.1
        self['wall_grasp']['up_speed'] = 0.1
        self['wall_grasp']['down_speed'] = 0.1
        self['wall_grasp']['wall_force'] = 10.0
        self['wall_grasp']['angle_of_attack'] = 1.0  # radians
        self['wall_grasp']['object_lift_time'] = 4.5

        self['edge_grasp']['initial_goal'] = np.array(
            [-0.05864322834179703, 0.4118988657714642, -0.05864200146127985, -1.6887810963180838, -0.11728653060066829,
             -0.8237944986945402, 0])
        self['edge_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.3])
        self['edge_grasp']['hand_object_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.05]),
                                                                          tra.rotation_matrix(math.radians(10.),
                                                                                              [1, 0, 0]),
                                                                          tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['grasp_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, -0.05, 0]),
                                                                    tra.rotation_matrix(math.radians(10.), [1, 0, 0]),
                                                                    tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['postgrasp_pose'] = tra.translation_matrix([0, 0, -0.1])
        self['edge_grasp']['downward_force'] = 4.0
        self['edge_grasp']['sliding_speed'] = 0.04
        self['edge_grasp']['valve_pattern'] = (
        np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]]), np.array([[0, 3.0]] * 6))


class RBOHandP24_pulpyWAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)

        # Define generic object parameters for surface grasp
        self['surface_grasp']['object']['up_speed'] = 0.30

        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.09, 0, 0.0]), tra.rotation_matrix(math.radians(20.0), [0, 1, 0]))

        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, -0.05]),
            tra.rotation_matrix(math.radians(-15.), [0, 1, 0]))

        # Define generic object parameters for surface grasp
        self['wall_grasp']['object']['lift_dist'] = 0.13  # short lift after initial contact (before slide)

        # object specific parameters for apple
        self['surface_grasp']['apple'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['apple']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]), tra.rotation_matrix(math.radians(25.0), [0, 1, 0]))

        # object specific parameters for cucumber
        self['surface_grasp']['cucumber'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['cucumber']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.045, 0, 0.0]), tra.rotation_matrix(math.radians(40.0), [0, 1, 0]))

        self['surface_grasp']['cucumber']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, -0.14]),
            tra.rotation_matrix(math.radians(-70.), [0, 1, 0]))

        # object specific parameters for punnet
        self['surface_grasp']['punnet'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['punnet']['pre_grasp_velocity'] = np.array([0.12, 0.06])

        self['surface_grasp']['punnet']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.1, -0.02, -0.0]),
            tra.rotation_matrix(math.radians(35.0), [0, 1, 0]))  # <-- best so far

        self['surface_grasp']['punnet']['downward_force'] = 10  # important, as it helps to fix the object and allows
        # the hand to wrap around the punnet such that it is stable. With lower values the grasps were almost always all
        # failing because the hand wasn't spreading out enough.

        self['surface_grasp']['punnet']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, -0.0]),
            tra.rotation_matrix(math.radians(0.), [0, 1, 0]))
        # tra.rotation_matrix(math.radians(10.), [1, 0, 0]))

        # object specific parameters for punnet
        self['surface_grasp']['mango'] = self['surface_grasp']['object'].copy()

        self['surface_grasp']['mango']['pregrasp_transform'] = tra.concatenate_matrices(
            # tra.translation_matrix([-0.03, 0.0, 0.0]), tra.rotation_matrix(math.radians(35.0), [0, 1, 0])) # <-- best so far
            tra.translation_matrix([-0.06, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(35.0), [0, 1, 0]))  # <-- best so far

        self['wall_grasp']['cucumber'] = self['wall_grasp']['object'].copy()
        self['wall_grasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0, -0.14]),  # 23 cm above object, 15 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(22.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))
