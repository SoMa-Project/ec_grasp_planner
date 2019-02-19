#!/usr/bin/env python

import math
import copy
import itertools
import numpy as np
from tf import transformations as tra


# python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp wall_grasp wall_chewinggum
# python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp edge_grasp --edgedistance -0.007 edge_chewinggum/
# python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp surface_grasp test_folder

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


class BaseHandArm(dict):
    def __init__(self):
        super(dict, self).__init__()

        self['mesh_file'] = "Unknown"
        self['mesh_file_scale'] = 1.

        # The name of the supported strategies
        self.__strategy_names = ['wall_grasp', 'edge_grasp', 'surface_grasp']

        for strategy in self.__strategy_names:
            # every strategy defines a default parameter set called 'object'
            self[strategy] = {'object': {}}

        self['isForceControllerAvailable'] = False

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

        # use null space controller to avoid joint limits during execution
        self['use_null_space_posture'] = True  # TODO Disney: implement this controller or set to False

        # max waiting time to trigger hand over, otherwise drop off object
        self['wait_handing_over_duration'] = 8
        # SURFACE GRASP
        # ---------------------------
        # Generic Object
        # ---------------------------

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
            tra.translation_matrix([-0.02, 0, 0.0]), tra.rotation_matrix(math.radians(25.0), [0, 1, 0]))

        # first motion after grasp, in hand palm frame
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(-10.),
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
            [0.146824, 0.948542, -0.135149, 1.85859, -1.75554, -0.464148, 1.53275])  # in the center of the table
        #   [-0.57148, 0.816213, -0.365673, 1.53765, 0.30308, 0.128965, 1.02467])    # next to the table

        # object hand over configuration - this is system specific!
        self['surface_grasp']['object']['hand_over_config'] = np.array(
            [0.650919, 1.04026, -0.940386, 1.30763, 0.447859, 0.517442, 0.0633935])

        # synergy type for soft hand closing
        self['surface_grasp']['object']['hand_closing_synergy'] = 0  # TODO check that

        # time of soft hand closing
        self['surface_grasp']['object']['hand_closing_duration'] = 5

        # Workaround to prevent joint limits (don't start at view position) TODO get rid of that!
        self['surface_grasp']['object']['initial_goal'] = np.array(
            [0.457199, 0.506295, -0.268382, 2.34495, 0.168263, 0.149603, -0.332482])  # different for ocado use-case

        # time of soft hand closing
        self['surface_grasp']['object']['down_dist'] = 0.5
        self['surface_grasp']['object']['up_dist'] = 0.12  # This is different compared to ocado use-case
        self['surface_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.06])  # first value: rotational, second translational

        # maximal joint velocities in case a JointController is used (e.g. alternative behavior was generated)
        self['surface_grasp']['object']['pre_grasp_joint_velocity'] = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
        self['surface_grasp']['object']['go_down_joint_velocity'] = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])  #np.ones(7) * 0.2

        # the force with which the person pulls the object out of the hand
        self['surface_grasp']['object']['hand_over_force'] = 2.5

        self['surface_grasp']['object']['pre_grasp_velocity'] = np.array([0.125, 0.08])

        # defines the manifold in which alternative goal poses are sampled during kinematic checks
        # for object specific ones look further down.
        self['surface_grasp']['object']['pre_grasp_manifold'] = Manifold({'min_position_deltas': [-0.05, -0.05, -0.05],#[-0.01, -0.01, -0.01],
                                                                          'max_position_deltas': [0.05, 0.05, 0.05],#[0.01, 0.01, 0.01],
                                                                          'min_orientation_deltas': [0, 0, 0],#-1.5],
                                                                          'max_orientation_deltas': [0, 0, 0],#1.5]
                                                                         })

        self['surface_grasp']['object']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.05],
                                                                        'max_position_deltas': [0.06, 0.04, 0.01],
                                                                        'min_orientation_deltas': [0, 0, 0],
                                                                        'max_orientation_deltas': [0, 0, 0]
                                                                       })

        self['surface_grasp']['object']['post_grasp_rot_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.05],
                                                                               'max_position_deltas': [0.06, 0.04, 0.01],
                                                                               'min_orientation_deltas': [0, 0, 0],
                                                                               'max_orientation_deltas': [0, 0, 0]
                                                                              })

        self['surface_grasp']['object']['go_up_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.05],
                                                                      'max_position_deltas': [0.06, 0.04, 0.01],
                                                                      'min_orientation_deltas': [0, 0, 0],
                                                                      'max_orientation_deltas': [0, 0, 0]
                                                                      })

        self['surface_grasp']['object']['go_drop_off_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.05],
                                                                            'max_position_deltas': [0.06, 0.04, 0.01],
                                                                            'min_orientation_deltas': [0, 0, 0],
                                                                            'max_orientation_deltas': [0, 0, 0]
                                                                            })

        # SURFACE GRASP
        # ----------------------------------------------------------------------------
        # Specific Objects: plushtoy, apple, egg, headband, bottle, banana, ticket
        # ----------------------------------------------------------------------------

        self['surface_grasp']['plushtoy'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['plushtoy']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(0.),
                                [0, 1, 0]))

        self['surface_grasp']['plushtoy']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.08],
                                                                        'max_position_deltas': [0.06, 0.04, 0.01],
                                                                        'min_orientation_deltas': [0, 0, 0],
                                                                        'max_orientation_deltas': [0, 0, 0]
                                                                       })

        #drop configuration - this is system specific!
        self['surface_grasp']['apple'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['egg'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['headband'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['bottle'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['banana'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['ticket'] = self['surface_grasp']['object'].copy()

        self['surface_grasp']['apple']['hand_over_config'] = np.array(
            [0.643723, 1.08375, -0.731847, 1.80354, -1.96563, 0.890579, 0.295289])
        self['surface_grasp']['egg']['hand_over_config'] = np.array(
            [0.19277, 0.938904, -0.206532, 1.52452, -2.57598, -0.0341588, 2.65164])
        self['surface_grasp']['headband']['hand_over_config'] = np.array(
            [-0.122134, 1.04449, 0.384282, 1.48404, 0.256033, -1.32681, 2.31987])
        self['surface_grasp']['bottle']['hand_over_config'] = np.array(
            [0.643723, 1.08375, -0.731847, 1.80354, -1.96563, 0.890579, 0.295289])

        self['surface_grasp']['banana']['hand_over_config'] = np.array(
            [-0.109826, 1.05006, 0.353494, 1.88186, 0.252395, -1.23794, 1.80944])

        self['surface_grasp']['banana']['hand_over_force'] = 5.0

        #####################################################################################
        # below are parameters for edge grasp with P24 fingers (standard RBO hand)
        #####################################################################################


        # EDGE GRASP
        # ----------------------------------------------------------------------------
        # Specific Objects: headband, ticket
        # ----------------------------------------------------------------------------

        #synergy type for soft hand closing
        self['edge_grasp']['object']['hand_closing_synergy'] = 0
        self['edge_grasp']['object']['hand_closing_duration'] = 5
        self['edge_grasp']['object']['initial_goal'] = np.array(
            [0.764798, 1.11152, -1.04516, 2.09602, -0.405398, -0.191906, 2.01431])
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
            tra.translation_matrix([-0.08, 0, -0.23]),  # 23 cm above object
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(35.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        # first motion after grasp, in hand palm frame
        self['edge_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),  # nothing right now
            tra.rotation_matrix(math.radians(0.0),
                                [0, 1, 0]))

        #drop configuration - this is system specific!
        self['edge_grasp']['object']['drop_off_config'] = np.array(
            [0.600302, 0.690255, 0.00661675, 2.08453, -0.0533508, -0.267344, 0.626538])

        # object hand over configuration - this is system specific!
        self['edge_grasp']['object']['hand_over_config'] = np.array(
            [0.650919, 1.04026, -0.940386, 1.30763, 0.447859, 0.517442, 0.0633935])

        # the force with which the person pulls the object out of the hand
        self['edge_grasp']['object']['hand_over_force'] = 2.5
        self['edge_grasp']['object']['table_force'] = 3.0
        self['edge_grasp']['object']['up_dist'] = 0.2
        self['edge_grasp']['object']['down_dist'] = 0.25
        self['edge_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.03])  # first value: rotational, second translational
        self['edge_grasp']['object']['slide_velocity'] = np.array([0.125, 0.03])
        self['edge_grasp']['object']['palm_edge_offset'] = 0


        # EDGE GRASP
        # ----------------------------------------------------------------------------
        # Specific Objects: headband, ticket
        # ----------------------------------------------------------------------------

        #drop configuration - this is system specific!
        self['edge_grasp']['headband'] = self['edge_grasp']['object'].copy()
        self['edge_grasp']['ticket'] = self['edge_grasp']['object'].copy()
        self['edge_grasp']['plushtoy'] = self['edge_grasp']['object'].copy()
        self['edge_grasp']['apple'] = self['edge_grasp']['object'].copy()
        self['edge_grasp']['egg'] = self['edge_grasp']['object'].copy()
        self['edge_grasp']['bottle'] = self['edge_grasp']['object'].copy()
        self['edge_grasp']['banana'] = self['edge_grasp']['object'].copy()

        self['edge_grasp']['apple']['hand_over_config'] = np.array(
            [0.643723, 1.08375, -0.731847, 1.80354, -1.96563, 0.890579, 0.295289])
        self['edge_grasp']['egg']['hand_over_config'] = np.array(
            [0.19277, 0.938904, -0.206532, 1.52452, -2.57598, -0.0341588, 2.65164])
        self['edge_grasp']['headband']['hand_over_config'] = np.array(
            [-0.122134, 1.04449, 0.384282, 1.48404, 0.256033, -1.32681, 2.31987])
        self['edge_grasp']['bottle']['hand_over_config'] = np.array(
            [0.643723, 1.08375, -0.731847, 1.80354, -1.96563, 0.890579, 0.295289])
        self['edge_grasp']['banana']['hand_over_config'] = np.array(
            [-0.109826, 1.05006, 0.353494, 1.88186, 0.252395, -1.23794, 1.80944])
        self['edge_grasp']['plushtoy']['hand_over_config'] = np.array(
            [-0.109826, 1.05006, 0.353494, 1.88186, 0.252395, -1.23794, 1.80944])
        self['edge_grasp']['ticket']['hand_over_config'] = np.array(
            [-0.122134, 1.04449, 0.384282, 1.48404, 0.256033, -1.32681, 2.31987])



        # plush toy
        self['edge_grasp']['plushtoy']['table_force'] = 10.0
        self['edge_grasp']['plushtoy']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.03, 0, -0.23]),  # 23 cm above object
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.0), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(18.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))


        # head band
        self['edge_grasp']['headband']['table_force'] = 1.5
        self['edge_grasp']['headband']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.08, 0.0, -0.23]),  # 23 cm above object
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.0), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(34.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))
        self['edge_grasp']['headband']['palm_edge_offset'] = 0.06


        # ticket
        self['edge_grasp']['ticket']['table_force'] = 1.1
        self['edge_grasp']['ticket']['pre_approach_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.1, 0.03, -0.23]),  # 23 cm above object
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.0), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(35.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
                tra.rotation_matrix(  # this makes the fingers point downwards
                    math.radians(0.0), [0, 0, 1]),
            ))

        self['edge_grasp']['ticket']['palm_edge_offset'] = 0.1

        self['edge_grasp']['ticket']['hand_over_config'] = np.array(
            [-0.122134, 1.04449, 0.384282, 1.48404, 0.256033, -1.32681, 2.31987])

        self['edge_grasp']['ticket']['hand_over_force'] = 2.0 #open automatically



        #####################################################################################
        # WALL GRASP - not used in disney
        #####################################################################################
        self['edge_grasp']['object']['hand_closing_synergy'] = 0
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

        # the pre-approach pose should be:              # TODO maybe move a little in negative y direction!
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
            [-0.318009, 0.980688, -0.538178, 1.67298, -2.07823, 0.515781, -0.515471])

        self['wall_grasp']['object']['table_force'] = 1.5
        self['wall_grasp']['object']['lift_dist'] = 0.1  # short lift after initial contact (before slide)
        self['wall_grasp']['object']['sliding_dist'] = 0.4  # sliding distance, should be min. half Ifco size
        self['wall_grasp']['object']['up_dist'] = 0.2
        self['wall_grasp']['object']['down_dist'] = 0.25
        self['wall_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.06])  # first value: rotational, second translational
        self['wall_grasp']['object']['slide_velocity'] = np.array([0.125, 0.06])
        self['wall_grasp']['object']['wall_force'] = 3.0

        # maximal joint velocities in case a JointController is used (e.g. alternative behavior was generated)
        self['wall_grasp']['object']['max_joint_velocity'] = np.ones(7) * 0.2
        # maximal joint velocities during sliding motion in case a JointController is used.
        self['wall_grasp']['object']['slide_joint_velocity'] = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])

        # defines the manifold in which alternative goal poses are sampled during feasibility checks
        #self['wall_grasp']['object']['init_joint_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.01, -0.01],
        #                                                               'max_position_deltas': [0.01, 0.01, 0.01],
        #                                                               'min_orientation_deltas': [0, 0, -0.001],
        #                                                               'max_orientation_deltas': [0, 0, 0.001]
        #                                                               })

        self['wall_grasp']['object']['pre_grasp_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.01, -0.02],
                                                                     'max_position_deltas': [0.01, 0.01, 0.02],
                                                                     'min_orientation_deltas': [0, 0, -0.17],
                                                                     'max_orientation_deltas': [0, 0, 0.17]
                                                                     })

        self['wall_grasp']['object']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.02, -0.04],
                                                                     'max_position_deltas': [0.01, 0.02, 0.06],
                                                                     'min_orientation_deltas': [0, 0, -np.pi/16.0],
                                                                     'max_orientation_deltas': [0, 0, np.pi/16.0]
                                                                     })

        self['wall_grasp']['object']['lift_hand_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.02, -0.04],
                                                                       'max_position_deltas': [0.01, 0.02, 0.04],
                                                                       'min_orientation_deltas': [0, 0, -np.pi/16.0],
                                                                       'max_orientation_deltas': [0, 0, np.pi/16.0]
                                                                       })

        self['wall_grasp']['object']['slide_to_wall_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.01, -0.01],
                                                                           'max_position_deltas': [0.01, 0.01, 0.01],
                                                                           'min_orientation_deltas': [0, 0, -0.17],
                                                                           'max_orientation_deltas': [0, 0, 0.17]
                                                                          })


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
        self['wall_grasp']['sliding_dist'] = 0.1
        self['wall_grasp']['up_dist'] = 0.1
        self['wall_grasp']['down_dist'] = 0.1
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
        self['edge_grasp']['sliding_dist'] = 0.04
        self['edge_grasp']['valve_pattern'] = (
        np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]]), np.array([[0, 3.0]] * 6))


class RBOHandP24_pulpyWAM(RBOHandP24WAM):
    def __init__(self, **kwargs):
        RBOHandP24WAM.__init__(self, **kwargs)

        # Define generic object parameters for surface grasp
        self['surface_grasp']['object']['up_dist'] = 0.30

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

        # object specific parameters for mango
        self['surface_grasp']['mango'] = self['surface_grasp']['object'].copy()

        self['surface_grasp']['mango']['pregrasp_transform'] = tra.concatenate_matrices(
            # tra.translation_matrix([-0.03, 0.0, 0.0]), tra.rotation_matrix(math.radians(35.0), [0, 1, 0])) # <-- best so far
            tra.translation_matrix([-0.06, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(35.0), [0, 1, 0]))  # <-- best so far

        self['surface_grasp']['mango']['pre_grasp_manifold'] = Manifold(position_deltas=[0.04, 0.04, 0.04],
                                                                        orientation_deltas=[0, 0, np.pi])

        self['surface_grasp']['mango']['go_down_manifold'] =  Manifold({'min_position_deltas': [-0.09, -0.09, -0.09],
                                                                        'max_position_deltas': [0.09, 0.09, 0.09],

                                                                        'min_orientation_deltas': [0, 0,  -np.pi],
                                                                        'max_orientation_deltas': [0, 0,  np.pi]
                                                                       })



        #Manifold(position_deltas=[0.03, 0.05, 0.08],
                                                             #         orientation_deltas=[0, 0, np.pi])

        # object specific parameters for cucumber (wall grasp)
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
