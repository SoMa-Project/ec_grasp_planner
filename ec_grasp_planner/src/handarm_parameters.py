#!/usr/bin/env python

import math
import copy
import itertools
import numpy as np
from tf import transformations as tra


# python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp WallGrasp wall_chewinggum
# python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp EdgeGrasp --edgedistance -0.007 edge_chewinggum/
# python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp SurfaceGrasp test_folder

class Manifold(dict):
    def __init__(self, initializer_dict=None, position_deltas=None, orientation_deltas=None):
        super(dict, self).__init__()

        # Parse convenient parameters for symmetric manifolds (min == max)
        if position_deltas is not None:
            self['max_position_deltas'] = [max(-pd, pd) for pd in position_deltas]
            self['min_position_deltas'] = [min(-pd, pd) for pd in position_deltas]
        if orientation_deltas is not None:
            self['max_orientation_deltas'] = [max(-od, od) for od in orientation_deltas]
            self['min_orientation_deltas'] = [min(-od, od) for od in orientation_deltas]

        # parse (and overwrite) any paramters given in the initializer list
        if initializer_dict:
            for k in initializer_dict.keys():
                self[k] = copy.copy(initializer_dict[k])

    def check_if_valid(self):

        if not len(self) == 4:
            AssertionError("Manifold: Illegal Number of attributes. Expected 4, but was {}".format(len(self)))

        required_keys = ['min_position_deltas', 'max_position_deltas', 'min_orientation_deltas',
                         'min_orientation_deltas']
        for k in required_keys:
            if k not in self:
                AssertionError("Manifold: Required key {} missing".format(k))

            if len(self[k]) != 3:
                AssertionError("Manifold: Key {0} has wrong number of coordinates. Expected 3, but was {1}".format(
                    k, len(self[k])))

        return True

    def __copy__(self):
        # ensures that every copy is automatically also a deep copy
        return copy.deepcopy(self)

    @staticmethod
    def deg_to_rad(angles_degree):
        return [ad * np.pi / 180.0 for ad in angles_degree]


class BaseHandArm(dict):
    def __init__(self):
        super(dict, self).__init__()

        self['mesh_file'] = "Unknown"
        self['mesh_file_scale'] = 1.

        # The names of the supported strategies
        self.__strategy_names = ['WallGrasp', 'EdgeGrasp', 'SurfaceGrasp', 'CornerGrasp']

        for strategy in self.__strategy_names:
            # every strategy defines a default parameter set called 'object'
            self[strategy] = {'object': {}}

        self['success_estimator_timeout'] = 10

    def __copy__(self):
        # ensures that every copy is automatically also a deep copy
        return copy.deepcopy(self)

    def checkValidity(self):
        # This function should always be called after the constructor of any class inherited from BaseHandArm
        # This convenience function allows to combine multiple sanity checks to ensure the handarm_parameters are as intended.
        self.assertNoCopyMissing()
        self.assert_manifolds_valid()

    def assertNoCopyMissing(self):
        for s, s_other in itertools.product(self.__strategy_names, repeat=2):
            for obj in self[s]:
                for obj_other in self[s_other]:
                    if not obj_other == obj and self[s][obj] is self[s_other][obj_other]:
                        # unintended reference copy of dictionary.
                        # This probably means that some previously defined parameters were overwritten.
                        raise AssertionError("You probably forgot to call copy(): {0} and {1} are equal for {2}".format(
                            obj, obj_other, s))

    def assert_manifolds_valid(self):
        for strategy in self.__strategy_names:
            for obj in self[strategy]:
                for param in self[strategy][obj]:
                    if param.endswith("manifold"):
                        self[strategy][obj][param].check_if_valid()


# ----------------------------------------------------------------- #
# --------------- Parameter Definitions for WAM arm --------------- #
# ----------------------------------------------------------------- #
class WAM(BaseHandArm):
    def __init__(self):
        super(WAM, self).__init__()

        # This defines the robot noise distribution for the grasp success estimator, as calculated by
        # calculate_success_estimator_object_params.py. First value is mean, second is standard deviation.
        # This is mainly robot specific, but depending on the accuracy of the hand models each hand might introduce
        # additional noise. In that case the values should be updated in their specific classes
        self['success_estimation_robot_noise'] = np.array([0.0323, 0.0151])


class RBOHand2(WAM):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1
        self['drop_off_config'] = np.array([-0.57148, 0.816213, -0.365673, 1.53765, 0.30308, 0.128965, 1.02467])
# This map defines all grasp parameter such as poses and configurations for a specific robot system


# Define this map for your system if you want to port the planner
# Rbo hand 2 (P24 fingers and rotated palm) mounted on WAM.
class RBOHandP24WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHandP24WAM, self).__init__()

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        # Workaround to prevent reaching joint limits (don't start at view position) TODO get rid of that!
        self['SurfaceGrasp']['object']['initial_goal'] = np.array(
            [0.387987, 0.638624, -0.361978, 2.10522, -0.101053, -0.497832, -0.487216])

        # transformation between object frame and hand palm frame above the object- should not be changed per object
        # please don't set x and y position, this should be done in pre_grasp_transform
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.3]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(90.), [0, 0, 1]),
                tra.rotation_matrix(
                    math.radians(180.), [1, 0, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # ---- Pre-approach parameters ----
        # position of hand relative to the object before and at grasping
        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]), tra.rotation_matrix(math.radians(10.0), [0, 1, 0]))

        # Maximum velocity of the EE during the pre approach movement. First value: rotational, second translational
        self['SurfaceGrasp']['object']['pre_approach_velocity'] = np.array([0.125, 0.08])

        # Maximal joint velocities in case a JointController is used (e.g. alternative behavior was genererated)
        self['SurfaceGrasp']['object']['pre_approach_joint_velocity'] = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])

        # Defines the general manifold in which alternative goal poses are sampled during kinematic checks.
        # You can also define special manifolds per obejct
        self['SurfaceGrasp']['object']['pre_approach_manifold'] = Manifold({'min_position_deltas': [-0.05, -0.05, -0.05],
                                                                          'max_position_deltas': [0.05, 0.05, 0.05],
                                                                          'min_orientation_deltas': [0, 0, 0],
                                                                          'max_orientation_deltas': [0, 0, 0],
                                                                         })

        # ---- Go-Down parameters ----
        # The maximum allowed force for pushing down (guarding the downward movement)
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # Distance that should be moved on guarded move when approaching the object
        self['SurfaceGrasp']['object']['down_dist'] = 0.35

        # Maximum velocity of the EE during the go down movement. First value: rotational, second translational
        self['SurfaceGrasp']['object']['go_down_velocity'] = np.array([0.125, 0.09])

        # Maximal joint velocities in case a JointController is used (e.g. alternative behavior was genererated)
        self['SurfaceGrasp']['object']['go_down_joint_velocity'] = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])

        # Defines the general manifold in which alternative goal poses are sampled during kinematic checks.
        # You can also define special manifolds per obejct
        self['SurfaceGrasp']['object']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.05],
                                                                        'max_position_deltas': [0.06, 0.04, 0.01],
                                                                        'min_orientation_deltas': [0, 0, 0],
                                                                        'max_orientation_deltas': [0, 0, 0]
                                                                       })

        # ---- Hand-closing parameters ----
        # synergy type for soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_synergy'] = 0

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 5

        # ---- Lifting parameters ----
        # first motion after grasp, in hand palm frame
        self['SurfaceGrasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(-15.),
                                [0, 1, 0]))

        # Distance that the hand should be lifted after grasping the object
        self['SurfaceGrasp']['object']['up_dist'] = 0.409

        #####################################################################################
        # below are parameters for wall grasp with P24 fingers (standard RBO hand)
        #####################################################################################

        self['WallGrasp']['object']['initial_goal'] = np.array(
            [0.258841, 0.823679, -0.00565591, 1.67988, -0.87263, 0.806526, -1.03372])

        # Maximal joint velocities in case a JointController is used (e.g. alternative behavior was generated)
        # This is the general maximal velocity for any joint controller during a wall grasp.
        # For the sliding motion however, we define different maximal joint velocities (see: slide_joint_velocity)
        self['WallGrasp']['object']['max_joint_velocity'] = np.ones(7) * 0.2

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

        # ---- Pre-approach parameters ----
        # the pre-approach pose should be:
        # - floating above and behind the object,
        # - fingers pointing downwards
        # - palm facing the object and wall
        # This is what should be changed per object if needed...
        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0, -0.14]),  # 23 cm above object, 14 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(15.0), [0, 1, 0]),  # hand rotated 15 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        # Defines the general manifold for pre_approach in which alternative goal poses are sampled during kinematics
        # checks. You can also define special manifolds per obejct
        self['WallGrasp']['object']['pre_approach_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.01, -0.02],
                                                                     'max_position_deltas': [0.01, 0.01, 0.02],
                                                                     'min_orientation_deltas': [0, 0, -0.17],
                                                                     'max_orientation_deltas': [0, 0, 0.17]
                                                                     })

        # ---- Go-Down parameters ----
        # The maximum allowed force for pushing down (guarding the movement to the ifco bottom)
        self['WallGrasp']['object']['downward_force'] = 1.8

        # Distance that should be moved on guarded move when going behind the object
        self['WallGrasp']['object']['down_dist'] = 0.25

        # Maximum velocity of the EE during the go down movement. First value: rotational, second translational
        self['WallGrasp']['object']['go_down_velocity'] = np.array([0.125, 0.09])

        # Defines the general manifold for go_down in which alternative goal poses are sampled during kinematics
        # checks. You can also define special manifolds per obejct
        self['WallGrasp']['object']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.02, -0.04],
                                                                     'max_position_deltas': [0.01, 0.02, 0.06],
                                                                     'min_orientation_deltas': [0,         0, -np.pi / 16.0],
                                                                     'max_orientation_deltas': [0, np.pi / 2,  np.pi / 16.0]
                                                                     })

        # ---- Sliding parameters ----
        # Short lift after initial contact with bottom (before slide)
        self['WallGrasp']['object']['corrective_lift_dist'] = 0.1

        # Defines the general manifold for corrective_lift in which alternative goal poses are sampled during kinematics
        # checks. You can also define special manifolds per obejct
        self['WallGrasp']['object']['corrective_lift_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.02, -0.04],
                                                                       'max_position_deltas': [0.01, 0.02, 0.04],
                                                                       'min_orientation_deltas': [0, 0, -np.pi/16.0],
                                                                       'max_orientation_deltas': [0, 0, np.pi/16.0]
                                                                       })

        # The maximum allowed force for pushing against the wall (guarding the sliding movement)
        self['WallGrasp']['object']['wall_force'] = 12.0

        # Sliding distance. Should be at least half the ifco size
        self['WallGrasp']['object']['sliding_dist'] = 0.4

        # Maximum velocity of the EE during the sliding movement. First value: rotational, second translational
        self['WallGrasp']['object']['slide_velocity'] = np.array([0.125, 0.30])  # np.array([0.125, 0.12])

        # Defines the general manifold for sliding motion in which alternative goal poses are sampled during kinematics
        # checks. You can also define special manifolds per obejct
        self['WallGrasp']['object']['slide_to_wall_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.01, -0.01],
                                                                           'max_position_deltas': [0.01, 0.01, 0.01],
                                                                           'min_orientation_deltas': [0, 0, -0.17],
                                                                           'max_orientation_deltas': [0, 0, 0.17]
                                                                          })

        # ---- Hand closing ----
        # synergy type for soft hand closing
        self['WallGrasp']['object']['hand_closing_synergy'] = 0

        # time of soft hand closing
        self['WallGrasp']['object']['hand_closing_duration'] = 1.0

        # ---- Drop off parameters ----
        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.05, 0.0, 0.0]),  # nothing right now
            tra.rotation_matrix(math.radians(-18.0),
                                [0, 1, 0]))

        # Distance that the hand should be lifted after grasping the object
        self['WallGrasp']['object']['up_dist'] = 0.25

        # Maximal joint velocities during sliding motion in case a JointController is used.
        # (e.g. alternative behavior was generated)
        self['WallGrasp']['object']['slide_joint_velocity'] = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.15])


        #####################################################################################
        # below are parameters for corner grasp with P24 fingers (standard RBO hand)
        #####################################################################################

        self['CornerGrasp']['object']['initial_goal'] = np.array(
            [0.458148, 0.649566, -0.30957, 2.22163, -1.88134, 0.289638, -0.326112])

        # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
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

        # ---- Pre-approach parameters ----
        # the pre-approach pose should be:
        # - floating above and behind the object,
        # - fingers pointing downwards
        # - palm facing the object and wall
        self['CornerGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0.0, -0.14]),  # 23 cm above object, 15 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(15.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        # ---- Go-Down parameters ----
        # The maximum allowed force for pushing down (guarding the movement to the ifco bottom)
        self['CornerGrasp']['object']['downward_force'] = 1.8
        # Distance that should be moved on guarded move when going behind the object
        self['CornerGrasp']['object']['down_dist'] = 0.25
        # Maximum velocity of the EE during the go down movement. First value: rotational, second translational
        self['CornerGrasp']['object']['go_down_velocity'] = np.array([0.125, 0.09])

        # ---- Sliding parameters ----
        # Short lift after initial contact with bottom (before slide)
        self['CornerGrasp']['object']['corrective_lift_dist'] = 0.1

        # The maximum allowed force for pushing against the wall (guarding the sliding movement)
        self['CornerGrasp']['object']['wall_force'] = 12.0 # aggressive 17

        # Sliding distance. Should be at least half the ifco size
        self['CornerGrasp']['object']['sliding_dist'] = 0.4

        # Maximum velocity of the EE during the sliding movement. First value: rotational, second translational
        self['CornerGrasp']['object']['slide_velocity'] = np.array([0.125, 0.3]) #.063 for empty tennis balls; aggresive 0.5 # np.array([0.125, 0.12])

        # ---- Hand closing ----
        # synergy type for soft hand closing
        self['CornerGrasp']['object']['hand_closing_synergy'] = 0

        # time of soft hand closing
        self['CornerGrasp']['object']['hand_closing_duration'] = 2.0

        # ---- Drop off parameters ----
        # first motion after grasp, in hand palm frame
        self['CornerGrasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.02, -0.0, 0.0]),  # nothing right now
            tra.rotation_matrix(math.radians(-10.0),
                                 [0, 1, 0]))

        # Distance that the hand should be lifted after grasping the object
        self['CornerGrasp']['object']['up_dist'] = 0.25


class RBOHandP11WAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)


class RBOHandP24_opposableThumbWAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)


# Rbo hand 2 (Ocado version with long fingers and rotated palm) mounted on WAM.
class RBOHandO2WAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        super(RBOHandO2WAM, self).__init__()

        # This setup can grasp Ocado an punnet form IFCO
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

        #####################################################################################
        # Specific Parameters for Surface Grasp + P24_pulpy
        #####################################################################################

        # Define generic object parameters for surface grasp
        self['SurfaceGrasp']['object']['up_dist'] = 0.30

        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.09, 0, 0.0]), tra.rotation_matrix(math.radians(20.0), [0, 1, 0]))

        self['SurfaceGrasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, -0.05]),
            tra.rotation_matrix(math.radians(-15.), [0, 1, 0]))

        # Define generic object parameters for SurfaceGrasp
        # short lift after initial contact (before slide)
        self['WallGrasp']['object']['lift_dist'] = 0.13

        # object specific parameters for apple
        self['SurfaceGrasp']['apple'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['apple']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]), tra.rotation_matrix(math.radians(25.0), [0, 1, 0]))

        # object specific parameters for cucumber
        self['SurfaceGrasp']['cucumber'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.045, 0, 0.0]), tra.rotation_matrix(math.radians(40.0), [0, 1, 0]))

        self['SurfaceGrasp']['cucumber']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, -0.14]),
            tra.rotation_matrix(math.radians(-70.), [0, 1, 0]))

        # object specific parameters for punnet
        self['SurfaceGrasp']['punnet'] = self['SurfaceGrasp']['object'].copy()
        self['SurfaceGrasp']['punnet']['pre_approach_velocity'] = np.array([0.12, 0.06])

        self['SurfaceGrasp']['punnet']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.1, -0.02, -0.0]),
            tra.rotation_matrix(math.radians(35.0), [0, 1, 0]))  # <-- best so far

        self['SurfaceGrasp']['punnet']['downward_force'] = 10  # important, as it helps to fix the object and allows
        # the hand to wrap around the punnet such that it is stable. With lower values the grasps were almost always all
        # failing because the hand wasn't spreading out enough.

        self['SurfaceGrasp']['punnet']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, -0.0]),
            tra.rotation_matrix(math.radians(0.), [0, 1, 0]))
        # tra.rotation_matrix(math.radians(10.), [1, 0, 0]))

        # object specific parameters for mango
        self['SurfaceGrasp']['mango'] = self['SurfaceGrasp']['object'].copy()

        self['SurfaceGrasp']['mango']['downward_force'] = 5

        self['SurfaceGrasp']['mango']['pre_approach_transform'] = tra.concatenate_matrices(
            # tra.translation_matrix([-0.03, 0.0, 0.0]), tra.rotation_matrix(math.radians(35.0), [0, 1, 0])) # <-- best so far
            tra.translation_matrix([-0.07, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(35.0), [0, 1, 0]))  # <-- best so far

        self['SurfaceGrasp']['mango']['pre_approach_manifold'] = Manifold(position_deltas=[0.04, 0.04, 0.04],
                                                                          orientation_deltas=[0, 0, np.pi])

        self['SurfaceGrasp']['mango']['go_down_manifold'] =  Manifold({'min_position_deltas': [-0.09, -0.09, -0.09],
                                                                       'max_position_deltas': [0.09, 0.09, 0.09],
                                                                       'min_orientation_deltas': [0, 0,  -np.pi],
                                                                       'max_orientation_deltas': [0, 0,  np.pi]
                                                                       })

        #####################################################################################
        # Specific Parameters for Wall Grasp + P24_pulpy
        #####################################################################################

        # object specific parameters for cucumber (wall grasp)
        self['WallGrasp']['cucumber'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0, -0.14]),  # 23 cm above object, 15 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(22.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['mango'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['mango']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0.01, -0.15]),  # 23 cm above object, 15 cm behind, 1cm to the left
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(15.0), [0, 1, 0]),  # hand rotated 15 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        #####################################################################################
        # Specific Parameters for Corner Grasp + P24_pulpy
        #####################################################################################

        # Define generic object parameters for surface grasp
        self['CornerGrasp']['object']['lift_dist'] = 0.13  # short lift after initial contact (before slide)

        self['CornerGrasp']['cucumber'] = self['WallGrasp']['object'].copy()
        self['CornerGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
                tra.translation_matrix([-0.23, 0, -0.14]), #23 cm above object, 15 cm behind
                tra.concatenate_matrices(
                    tra.rotation_matrix(
                        math.radians(0.), [1, 0, 0]),
                    tra.rotation_matrix(
                        math.radians(22.0), [0, 1, 0]), #hand rotated 30 degrees on y = thumb axis
                    tra.rotation_matrix(                #this makes the fingers point downwards
                        math.radians(0.0), [0, 0, 1]),
            ))


# ----------------------------------------------------------------- #
# --------------- Parameter Definitions for KUKA arm -------------- #
# ----------------------------------------------------------------- #
class KUKA(BaseHandArm):
    def __init__(self, **kwargs):
        super(KUKA, self).__init__()

        # This defines the robot noise distribution for the grasp success estimator, as calculated by
        # calculate_success_estimator_object_params.py. First value is mean, second is standard deviation.
        # This is mainly robot specific, but depending on the accuracy of the hand models each hand might introduce
        # additional noise. In that case the values should be updated in their specific classes
        self['success_estimation_robot_noise'] = np.array([-0.0036, 0.04424])


class RBOHand2KUKA(KUKA):
    def __init__(self, **kwargs):
        super(RBOHand2KUKA, self).__init__()


class RBOHandO2KUKA(KUKA):
    def __init__(self, **kwargs):
        super(RBOHandO2KUKA, self).__init__()

        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"

        self['mesh_file_scale'] = 0.1

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]),
                                                         tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

        # duration of placing the object
        self['place_duration'] = 5

        self['place_down_speed'] = 0.05

        # This should be the same for all objects
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.3]),
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
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

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

        #####################################################################################
        # below are parameters for wall grasp with P24 fingers (standard RBO hand)
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
            tra.translation_matrix([-0.23, 0, -0.14]),  # 23 cm above object, 14 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.05

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.05

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.05  # sliding speed

        self['WallGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['WallGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['WallGrasp']['object']['hand_closing_duration'] = 1.0

        self['WallGrasp']['object']['hand_closing_synergy'] = 1

        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 2

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8

        # modify grasp parameters for cucumber
        # TODO: This is just an example...
        self['WallGrasp']['cucumber'] = self['WallGrasp']['object'].copy()
        self['WallGrasp']['cucumber']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0, -0.1]),  # 23 cm above object, 10 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))


class PISAHandKUKA(KUKA):
    def __init__(self, **kwargs):
        super(PISAHandKUKA, self).__init__()

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]),
                                                         tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

        # duration of placing the object
        self['place_duration'] = 5

        self['place_down_speed'] = 0.05

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

        self['SurfaceGrasp']['object']['hand_preshape_goal'] = 0.3

        # This should be the same for all objects
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.3]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(-90.), [0, 0, 1]),
                tra.rotation_matrix(
                    math.radians(180.), [1, 0, 0])))

        # transformation between the control frame of the hand and the frame in which the hand transform is defined
        # this is needed for the PISA hand to enforce the grasping signature
        # This should be the same for all objects
        # TODO: Change this to reflect the hand signature
        self['SurfaceGrasp']['object']['ee_in_goal_frame'] = tra.translation_matrix([0.0, 0.0, 0.0])

        # This is what should be changed per object if needed...
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.05

        self['SurfaceGrasp']['object']['up_speed'] = 0.05

        # synergy type for soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_goal'] = 0.8

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 5

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 8

        #####################################################################################
        # below are parameters for wall grasp
        #####################################################################################
        self['WallGrasp']['object']['hand_preshape_goal'] = 0.3

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
            tra.translation_matrix([-0.23, 0, -0.14]),  # 23 cm above object, 15 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.05

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.05

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.05  # sliding speed

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

        self['drop_off_pose'] = tra.concatenate_matrices(tra.translation_matrix([0.58436, 0.55982, 0.38793]),
                                                         tra.quaternion_matrix([0.95586, 0.27163, 0.10991, -0.021844]))

        # duration of placing the object
        self['place_duration'] = 5

        self['place_down_speed'] = 0.05

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

        self['SurfaceGrasp']['object']['hand_preshape_goal'] = 0.3

        # This should be the same for all objects
        self['SurfaceGrasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.3]),
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
        self['SurfaceGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['SurfaceGrasp']['object']['downward_force'] = 4

        # speed of approaching the object
        self['SurfaceGrasp']['object']['down_speed'] = 0.05

        self['SurfaceGrasp']['object']['up_speed'] = 0.05

        # synergy type for soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_goal'] = 0.8

        # time of soft hand closing
        self['SurfaceGrasp']['object']['hand_closing_duration'] = 5

        # duration of lifting the object
        self['SurfaceGrasp']['object']['lift_duration'] = 8

        #####################################################################################
        # below are parameters for wall grasp
        #####################################################################################
        self['WallGrasp']['object']['hand_preshape_goal'] = 0.3

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
        self['WallGrasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.23, 0, -0.14]),  # 23 cm above object, 15 cm behind
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        self['WallGrasp']['object']['downward_force'] = 1.8

        self['WallGrasp']['object']['down_speed'] = 0.05

        self['WallGrasp']['object']['corrective_lift_duration'] = 1.5

        self['WallGrasp']['object']['up_speed'] = 0.05

        self['WallGrasp']['object']['wall_force'] = 12.0

        self['WallGrasp']['object']['slide_speed'] = 0.05  # sliding speed

        self['WallGrasp']['object']['pre_grasp_twist'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self['WallGrasp']['object']['pre_grasp_rotation_duration'] = 0

        self['WallGrasp']['object']['hand_closing_duration'] = 1.0

        self['WallGrasp']['object']['hand_closing_goal'] = 0.8

        # first motion after grasp, in hand palm frame
        self['WallGrasp']['object']['post_grasp_twist'] = np.array([-0.05, 0.0, 0.0, 0.0, math.radians(-18.0), 0.0])

        self['WallGrasp']['object']['post_grasp_rotation_duration'] = 2

        # duration of lifting the object
        self['WallGrasp']['object']['lift_duration'] = 8

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