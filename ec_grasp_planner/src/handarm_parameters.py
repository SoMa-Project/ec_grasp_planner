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

    @staticmethod
    def deg_to_rad(angles_degree):
        return [ad * np.pi / 180.0 for ad in angles_degree]


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

class RBOHand2Prob(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2Prob, self).__init__()

        # use null space controller to avoid joint limits during execution
        self['use_null_space_posture'] = False  # TODO Disney: implement this controller or set to False

        # self['null_space_goal_is_relative'] = False # ??
        # self['surface_grasp']['object']['null_space_goal_is_relative'] = False

        # max waiting time to trigger hand over, otherwise drop off object
        self['wait_handing_over_duration'] = 8


        # This defines the robot noise distribution for the grasp success estimator, as calculated by
        # calculate_success_estimator_object_params.py. First value is mean, second is standard deviation.
        # This is mainly robot specific, but depending on the accuracy of the hand models each hand might introduce
        # additional noise. In that case the values should be updated in their specific classes
        ##TODO tune it
        self['success_estimation_robot_noise'] = np.array([0.0323, 0.0151])

        ##################  ADDED CODE NICOLAS ###################################
        # Generic Object
        # ---------------------------

        self['surface_grasp']['object']['max_joint_velocity'] = np.ones(6) * 0.2

        # you can define a default strategy for all objects by setting the second field to  'object'
        # for object-specific strategies set it to the object label

        self['surface_grasp']['object']['initial_goal'] = np.array([-0.13344680071507892, 0.3188774472606212, 0.8505352980103672, 0.1860369373347768, 1.866702739571019, -1.3398638725395275])
                #[-0.01, 0.4118988657714642, 1.32, 0.01, -0.4, 0]) without box, only table

        # transformation (only rotation) between object frame and hand palm frame
        # the convention at our lab is: x along the fingers and z normal on the palm.
        # please follow the same convention
        # self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
        #     tra.translation_matrix([0.0, 0.0, 0.0]),
        #     tra.concatenate_matrices(
        #         tra.rotation_matrix(
        #             math.radians(90.0), [1, 0, 0]),
        #         tra.rotation_matrix(
        #             math.radians(0.0), [0, 1, 0]),
        #         tra.rotation_matrix(
        #             math.radians(90.0), [0, 0, 1]),
        #     ))

        self['surface_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.15]),#0.3]),TODO: revert this back to 30cm above object
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(90.), [0, 0, 1]),
                tra.rotation_matrix(
                    math.radians(180.), [1, 0, 0])))

        # above the object, in hand palm frame
        # self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
        #     # tra.translation_matrix([-0.08, 0, 0.0]), tra.rotation_matrix(math.radians(25.0), [0, 1, 0]))
        #     # tra.translation_matrix([-0.04, 0, 0.0]), tra.rotation_matrix(math.radians(15.0), [0, 1, 0]))
        #     tra.translation_matrix([-0.08, 0, 0.0]), tra.rotation_matrix(math.radians(15.0), [0, 1, 0]))
        self['surface_grasp']['object']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, 0]),
            tra.concatenate_matrices(
                # tra.rotation_matrix(math.radians(-30), [0, 0, 1]),
                tra.rotation_matrix(math.radians(0), [0, 0, 1]),
                tra.rotation_matrix(math.radians(-20), [0, 1, 0])
            )
        )

        self['surface_grasp']['object']['pre_grasp_velocity'] = np.array([0.125, 0.08])

        # maximal joint velocities in case a JointController is used (e.g. alternative behavior was generated)
        self['surface_grasp']['object']['pre_grasp_joint_velocity'] = np.array([0.5]*6)
        self['surface_grasp']['object']['go_down_joint_velocity'] = np.array([0.2]*6)

        # first motion after grasp, in hand palm frame only rotation
        self['surface_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.rotation_matrix(math.radians(0.),
                                [0, 1, 0]))

        # second motion after grasp, in hand palm frame
        self['surface_grasp']['object']['go_up_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([-0.03, 0, -0.3]),
            tra.rotation_matrix(math.radians(-20.),
                                [0, 1, 0]))

        # the maximum allowed force for pushing down
        self['surface_grasp']['object']['downward_force'] = 4  # might be +10/-7 ??

        # drop configuration - this is system specific!
        self['surface_grasp']['object']['drop_off_config'] = np.array(
            # [0.600302, 0.690255, 0.00661675, 2.08453, -0.0533508, -0.267344]
            # [0.7854, 1.31186037, 1.676474819907, 0, -1.242999972232, 0])
            # [-1.0176, 1.7704, 0.5843, -1.8478, -1.9554, 0.8829])
            # this one is a bit close to the table: [-0.6847175734248546, 1.7276625360604194, 0.6593835433017187, 1.941366596533726, 1.396833792691509,
            # -2.335144021162548])
            [-0.6495122379513302, 1.0513008704056004, 1.1102978297770616, 1.8614173404205574, 1.2087920487824781,
             -2.2430899859608164])

        # object hand over configuration - this is system specific!
        self['surface_grasp']['object']['hand_over_config'] = np.array(
            [-0.5692783732632621, 0.6382217582392946, 1.0518671070385777, 1.3382843279658632, 0.4500335121370537,
             -2.9542495543962315])

        # the force with which the person pulls the object out of the hand
        self['surface_grasp']['object']['hand_over_force'] = 2.5



        # synergy type for soft hand closing
        self['surface_grasp']['object']['hand_closing_synergy'] = 0

        # time of soft hand closing
        self['surface_grasp']['object']['hand_closing_duration'] = 2

        # time of soft hand closing
        self['surface_grasp']['object']['down_speed'] = 0.5
        self['surface_grasp']['object']['up_speed'] = 0.25

        # TODO: this has been set to a small value since myP would otherwise complain about kinematic infeasibility
        # this currently depends on the value set for self['surface_grasp']['object']['hand_transform'] and is tuned
        # for alternative_behaviour from the feasibility module
        self['surface_grasp']['object']['down_dist'] = 0.05
        self['surface_grasp']['object']['down_dist_alt'] = 0.05
        self['surface_grasp']['object']['up_dist'] = 0.25
        self['surface_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.06])  # first value: rotational, second translational
        self['surface_grasp']['object']['pre_grasp_velocity'] = np.array([0.125, 0.08])

        # defines the manifold in which alternative goal poses are sampled during kinematic checks
        # for object specific ones look further down.
        self['surface_grasp']['object']['pre_grasp_manifold'] = Manifold(
            {'min_position_deltas': [-0.05, -0.05, -0.05],  # [-0.01, -0.01, -0.01],
             'max_position_deltas': [0.05, 0.05, 0.05],  # [0.01, 0.01, 0.01],
             'min_orientation_deltas': [0, 0, 0],  # -1.5],
             'max_orientation_deltas': [0, 0, 0],  # 1.5]
             })

        self['surface_grasp']['object']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.0, -0.0, -0.06],
                                                                        'max_position_deltas': [0.0, 0.0, -0.06],
                                                                        'min_orientation_deltas': [0, 0, 0],
                                                                        'max_orientation_deltas': [0, 0, 0]
                                                                        })

        self['surface_grasp']['object']['safety_distance_above_object'] = -0.03 #0

        self['surface_grasp']['object']['post_grasp_rot_manifold'] = Manifold(
            {'min_position_deltas': [-0.01, -0.04, -0.05],
             'max_position_deltas': [0.06, 0.04, 0.01],
             'min_orientation_deltas': [0, 0, 0],
             'max_orientation_deltas': [0, 0, 0]
             })

        self['surface_grasp']['object']['go_up_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.05],
                                                                      'max_position_deltas': [0.06, 0.04, 0.01],
                                                                      'min_orientation_deltas': [0, 0, 0],
                                                                      'max_orientation_deltas': [0, 0, 0]
                                                                      })

        self['surface_grasp']['object']['go_drop_off_manifold'] = Manifold(
            {'min_position_deltas': [-0.01, -0.04, -0.05],
             'max_position_deltas': [0.06, 0.04, 0.01],
             'min_orientation_deltas': [0, 0, 0],
             'max_orientation_deltas': [0, 0, 0]
             })

        self['surface_grasp']['apple'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['bottle'] = self['surface_grasp']['object'].copy()

        self['surface_grasp']['apple']['pre_grasp_manifold'] = Manifold(
            {'min_position_deltas': [-0.05, -0.05, -0.00],  # [-0.01, -0.01, -0.01],
             'max_position_deltas': [0.05, 0.05, 0.00],  # [0.01, 0.01, 0.01],
             'min_orientation_deltas': [0, 0, -np.pi],  # -1.5],
             'max_orientation_deltas': [0, 0, np.pi],  # 1.5]
             })

        self['surface_grasp']['apple']['go_down_manifold'] = Manifold(
            {'min_position_deltas': [0.0, 0.0, -0.00],  # [-0.01, -0.01, -0.01],
             'max_position_deltas': [0.0, 0.0, -0.00],  # [0.01, 0.01, 0.01],
             'min_orientation_deltas': [0, 0, -np.pi],  # -1.5],
             'max_orientation_deltas': [0, 0, np.pi],  # 1.5]
             })

        #########################################################################################################


        self['surface_grasp']['v_max'] = np.array([10] * 6)
        self['surface_grasp']['k_p'] = np.array([200, 150, 20, 10, 10, 5])
        self['surface_grasp']['k_v'] = np.array([10] * 6)

        self['surface_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.2])
        self['surface_grasp']['hand_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0]),
                                                                      tra.rotation_matrix(math.radians(0.), [0, 0, 1]))
        self['surface_grasp']['downward_force'] = 7.
        self['surface_grasp']['valve_pattern'] = (
        np.array([[0., 4.1], [0., 0.1], [0., 5.], [0., 5.], [0., 2.], [0., 3.5]]), np.array([[1, 0]] * 6))

        self['wall_grasp']['v_max'] = np.array([10] * 6)
        self['wall_grasp']['k_p'] = np.array([200, 150, 20, 10, 10, 5])
        self['wall_grasp']['k_v'] = np.array([10] * 6)
        self['wall_grasp']['pregrasp_pose'] = tra.translation_matrix([0.05, 0, -0.2])
        self['wall_grasp']['table_force'] = 7.
        self['wall_grasp']['sliding_speed'] = 0.1
        self['wall_grasp']['up_speed'] = 0.1
        self['wall_grasp']['down_speed'] = 0.1
        self['wall_grasp']['wall_force'] = 10.0
        self['wall_grasp']['angle_of_attack'] = 1.0  # radians
        self['wall_grasp']['object_lift_time'] = 4.5

        self['wall_grasp']['sliding_speed'] += 0

        self['edge_grasp']['v_max'] = np.array([10] * 6)
        self['edge_grasp']['k_p'] = np.array([200, 150, 20, 10, 10, 5])
        self['edge_grasp']['k_v'] = np.array([10] * 6)
        self['edge_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.3])
        self['edge_grasp']['hand_object_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.05]),
                                                                          tra.rotation_matrix(math.radians(10.),
                                                                                              [1, 0, 0]),
                                                                          tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['grasp_pose'] = tra.concatenate_matrices(tra.translation_matrix([0, -0.05, 0]),
                                                                    tra.rotation_matrix(math.radians(10.), [1, 0, 0]),
                                                                    tra.euler_matrix(0, 0, -math.pi / 2.))
        self['edge_grasp']['postgrasp_pose'] = tra.translation_matrix([0, 0, -0.1])
        self['edge_grasp']['downward_force'] = 7.0
        self['edge_grasp']['sliding_speed'] = 0.04
        self['edge_grasp']['valve_pattern'] = (
        np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]]), np.array([[0, 3.0]] * 6))




        # ------ EDGE GRASP --------------------------------------------------------------------------------------------

        self['surface_grasp']['object']['slide_joint_velocity'] = np.ones(6) * 0.2

        self['edge_grasp']['object'] = self['surface_grasp']['object'].copy()


        self['edge_grasp']['object']['hand_closing_synergy'] = 0
        self['edge_grasp']['object']['hand_closing_duration'] = 2
        self['edge_grasp']['object']['initial_goal'] = np.array([-0.13015969902674374, 0.27879386232497977, 0.9060628670263369, 0.1747674943188126, 1.7980381144022723, -1.4291233288800722])

        # self['edge_grasp']['object']['initial_goal'] = np.array([1.3198427865954807, 0.21985469987865527, 0.9847142456052655, -0.2950862646320009, 1.7794936535904256,
        #  1.4001143852676734])

        self['edge_grasp']['object']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(90.0), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(90.0), [0, 0, 1]),
            ))

        # self['edge_grasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
        #     tra.translation_matrix([-0.12, 0, -0.23]),  # 23 cm above object
        #     tra.concatenate_matrices(
        #         tra.rotation_matrix(
        #             math.radians(0.), [1, 0, 0]),
        #         tra.rotation_matrix(
        #             math.radians(5.0), [0, 1, 0]),  # hand rotated 30 degrees on y = thumb axis
        #         tra.rotation_matrix(  # this makes the fingers point downwards
        #             math.radians(0.0), [0, 0, 1]),
        #     ))

        self['edge_grasp']['object']['pre_approach_transform'] = tra.concatenate_matrices(
            # [0.07, 0.02, -0.10] 2019-04-07: these are gold standards if ticket has a good orientation
            tra.translation_matrix([0.0, 0.0, -0.10]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(-20.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 0, 1]),
            ))

        self['edge_grasp']['object']['pre_approach_transform_alt'] = tra.concatenate_matrices(
            # [0.07, 0.02, -0.10] 2019-04-07: these are gold standards if ticket has a good orientation
            # tra.translation_matrix([0.02, -0.015, -0.10]),
            # tra.translation_matrix([-0.02, 0.015, -0.10]),
            # tra.translation_matrix([-0.02, 0.0, -0.10]),
            # tra.translation_matrix([-0.02, 0.005, -0.10]),
            tra.translation_matrix([-0.01, 0.02, -0.10]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.0), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(-20.0), [0, 1, 0]),#math.radians(-20.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 0, 1]),
            ))

        # first motion after grasp, in hand palm frame
        self['edge_grasp']['object']['post_grasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),  # nothing right now
            tra.rotation_matrix(math.radians(0.0),
                                [0, 1, 0]))

        #drop configuration - this is system specific!
        self['edge_grasp']['object']['drop_off_config'] = self['surface_grasp']['object']['drop_off_config'].copy()

        # object hand over configuration - this is system specific!
        self['edge_grasp']['object']['hand_over_config'] = self['surface_grasp']['object']['hand_over_config'].copy()

        # the force with which the person pulls the object out of the hand
        self['edge_grasp']['object']['hand_over_force'] = 2.5
        self['edge_grasp']['object']['table_force'] = 3.0
        self['edge_grasp']['object']['up_dist'] = 0.1
        self['edge_grasp']['object']['down_dist'] = 0.25
        self['edge_grasp']['object']['go_down_velocity'] = np.array(
            [0.125, 0.03])  # first value: rotational, second translational
        self['edge_grasp']['object']['slide_velocity'] = np.array([0.125, 0.03])
        self['edge_grasp']['object']['palm_edge_offset'] = 0


        # EDGE GRASP
        # ----------------------------------------------------------------------------
        # Specific Objects: ticket
        # ----------------------------------------------------------------------------

        #drop configuration - this is system specific!
        self['edge_grasp']['ticket'] = self['edge_grasp']['object'].copy()


        self['edge_grasp']['ticket']['hand_over_config'] = np.array(
            [-0.5291471491727358, 0.9059347531337889, 0.8686134517585082, 0.7843957601103667, 1.280980273384745, -2.5911365575965055])

        # self['edge_grasp']['ticket']['palm_edge_offset'] = 0.03 # works well for conventional edge-grasp
        # self['edge_grasp']['ticket']['palm_edge_offset'] = -0.03
        self['edge_grasp']['ticket']['palm_edge_offset'] = 0.0

        self['edge_grasp']['ticket']['palm_edge_offset_alt'] = -0.01#-0.06


        self['edge_grasp']['ticket']['hand_over_force'] = 2.0 #open automatically

        # self['edge_grasp']['ticket']['post_slide_pose_trajectory'] = np.array([
        #     tra.translation_matrix([0, 0, -0.045]),
        #     tra.concatenate_matrices(tra.rotation_matrix(math.radians(35.0), [0, 1, 0]), tra.translation_matrix([0.04, 0.02, 0])),
        #     tra.translation_matrix([0, 0, 0.045]),
        #     np.eye(4), # tra.translation_matrix([-0.08, 0, 0])
        # ])

        self['edge_grasp']['ticket']['post_slide_pose_trajectory'] = np.array([
            tra.translation_matrix([-0.03, 0, -0.03]),
            # tra.translation_matrix([-0.02, 0, 0]),
            tra.translation_matrix([0, 0, 0.03]),
            tra.translation_matrix([0.03, 0, 0.0])
        ])

        self['edge_grasp']['ticket']['pre_grasp_manifold'] = Manifold(
            {'min_position_deltas': [-0.05, -0.05, -0.02],
             'max_position_deltas': [0.05, 0.05, 0.02],
             'min_orientation_deltas': [-np.pi / 16, -np.pi / 16, -np.pi / 4],
             'max_orientation_deltas': [np.pi / 16, np.pi / 16, np.pi / 4],
             })

        self['edge_grasp']['ticket']['go_down_manifold'] = Manifold(
            {'min_position_deltas': [-0.00, -0.00, -0.0],  # -0.05],
             'max_position_deltas': [0.00, 0.00, 0.0],  # 0.01],
             'min_orientation_deltas': [0, 0, -np.pi / 8.0],
             'max_orientation_deltas': [0, 0, np.pi / 8.0]
             })

        self['edge_grasp']['ticket']['slide_to_edge_manifold'] = Manifold(
            {'min_position_deltas': [-0.00, -0.00, -0.00],
             'max_position_deltas': [0.00, 0.00, 0.00],
             'min_orientation_deltas': [0, 0, -np.pi / 16.0],
             'max_orientation_deltas': [0, 0, np.pi / 16.0]
             })

        self['edge_grasp']['ticket']['slide_transform_alt'] = np.eye(4)

        self['edge_grasp']['ticket']['sliding_direction'] = 1


##################  ADDED CODE NICOLAS ###################################

class RBOHandP24_pulpyPROB(RBOHand2Prob):
    def __init__(self, **kwargs):
        RBOHand2Prob.__init__(self, **kwargs)


##################  PISA HAND  ###################################

class PisaIITHandProb(RBOHand2Prob):
    def __init__(self, **kwargs):
        RBOHand2Prob.__init__(self, **kwargs)

        self['success_estimation_robot_noise'] = np.array([0.91242, 0.019029])

        self['surface_grasp']['object']['hand_closing_duration'] = 0.5

        self['surface_grasp']['object']['down_dist_alt'] = 0.05

        self['surface_grasp']['object']['safety_distance_above_object'] = -0.03#0.05

        # for SH_V2
        # self['surface_grasp']['object']['safety_distance_above_object'] = 0.00


        self['surface_grasp']['apple'] = self['surface_grasp']['object'].copy()
        self['surface_grasp']['bottle'] = self['surface_grasp']['object'].copy()

        # above the object, in hand palm frame
        self['surface_grasp']['apple']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        self['surface_grasp']['bottle']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(0.0), [0, 1, 0]))

        self['surface_grasp']['apple']['pre_grasp_manifold'] = Manifold(
            {'min_position_deltas': [-0.05, -0.05, -0.05],  # [-0.01, -0.01, -0.01],
             'max_position_deltas': [0.05, 0.05, 0.05],  # [0.01, 0.01, 0.01],
             'min_orientation_deltas': [0, 0, -np.pi],  # -1.5],
             'max_orientation_deltas': [0, 0, np.pi],  # 1.5]
             })

        self['surface_grasp']['bottle']['pre_grasp_manifold'] = Manifold(
            {'min_position_deltas': [-0.05, -0.05, -0.0],
             'max_position_deltas': [0.05, 0.05, 0.0],
             'min_orientation_deltas': [-np.pi / 16, -np.pi / 16, -np.pi / 2],
             'max_orientation_deltas': [np.pi / 16, np.pi / 16, np.pi / 2],
             })

        self['surface_grasp']['apple']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.00, -0.00, -0.00],
                                                                       'max_position_deltas': [0.00, 0.00, -0.00],
                                                                       'min_orientation_deltas': [0, 0, -np.pi],
                                                                       'max_orientation_deltas': [0, 0, np.pi]
                                                                       })

        self['surface_grasp']['bottle']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.00, -0.00, -0.00],#-0.05],
                                                                       'max_position_deltas': [0.00, 0.00, -0.00],#0.01],
                                                                       'min_orientation_deltas': [0, 0, -np.pi/16.0],
                                                                       'max_orientation_deltas': [0, 0, np.pi/16.0]
                                                                       })

        self['surface_grasp']['banana'] = self['surface_grasp']['bottle'].copy()

        # self['surface_grasp']['banana']['pregrasp_transform'] = tra.concatenate_matrices(
        #     tra.translation_matrix([0.02, 0, 0]),
        #
        #     tra.rotation_matrix(math.radians(-10.0), [0, 1, 0]),
        #     tra.rotation_matrix(math.radians(-20.0), [1, 0, 0]),
        #     tra.rotation_matrix(math.radians(-20.0), [0, 0, 1])
        # )

        self['surface_grasp']['banana']['pregrasp_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.00, 0.03, 0]),

            tra.rotation_matrix(math.radians(-10.0), [0, 1, 0]),
            tra.rotation_matrix(math.radians(-10.0), [1, 0, 0]),
            tra.rotation_matrix(math.radians(-20.0), [0, 0, 1])
        )

        self['surface_grasp']['banana']['go_down_manifold'] = Manifold(
            {'min_position_deltas': [-0.03, -0.00, -0.00],  # -0.05],
             'max_position_deltas': [0.03, 0.00, -0.00],  # 0.01],
             'min_orientation_deltas': [0, 0, 0],
             'max_orientation_deltas': [0, 0, np.pi / 6.]
             })


        # ------ EDGE GRASP --------------------------------------------------------------------------------------------

        self['edge_grasp']['ticket'] = self['edge_grasp']['object'].copy()

        self['edge_grasp']['ticket']['palm_edge_offset'] = 0.0

        # TODO: maybe have a 'hand_transform' and 'hand_transform_alt' instead of 'pre_approach_transform' and
        # 'pre_approach_transform_alt'
        self['edge_grasp']['ticket']['pre_approach_transform'] = tra.concatenate_matrices(
            # [0.07, 0.02, -0.10] 2019-04-07: these are gold standards if ticket has a good orientation
            tra.translation_matrix([0.0, 0.0, -0.10]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(-20.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 0, 1]),
            ))


        self['edge_grasp']['ticket']['hand_closing_duration'] = 0.3

        self['edge_grasp']['ticket']['pre_grasp_manifold'] = Manifold(
            {'min_position_deltas': [-0.0, -0.0, -0.02],
             'max_position_deltas': [0.0, 0.0, 0.02],
             'min_orientation_deltas': [-np.pi / 16, -np.pi / 16, -np.pi / 4],
             'max_orientation_deltas': [np.pi / 16, np.pi / 16, np.pi / 4],
             })

        self['edge_grasp']['ticket']['go_down_manifold'] = Manifold(
            {'min_position_deltas': [-0.00, -0.00, -0.0],  # -0.05],
             'max_position_deltas': [0.00, 0.00, 0.0],  # 0.01],
             'min_orientation_deltas': [0, 0, -np.pi / 8.0],
             'max_orientation_deltas': [0, 0, np.pi / 8.0]
             })

        self['edge_grasp']['ticket']['slide_to_edge_manifold'] = Manifold(
            {'min_position_deltas': [-0.00, -0.00, -0.00],
             'max_position_deltas': [0.00, 0.00, 0.00],
             'min_orientation_deltas': [0, 0, -np.pi / 16.0],
             'max_orientation_deltas': [0, 0, np.pi / 16.0]
             })

        #
        self['edge_grasp']['ticket']['pre_approach_transform_alt'] = tra.concatenate_matrices(
            # tra.translation_matrix([0.025, 0.005, -0.10]),
            tra.translation_matrix([0.02, 0.0, -0.10]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(-20.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 0, 1]),
            ))

        # for SH_V2
        # self['edge_grasp']['ticket']['slide_transform_alt'] = tra.concatenate_matrices(
        #     tra.translation_matrix([0.0, 0.0, 0.0]),
        #     tra.concatenate_matrices(
        #         tra.rotation_matrix(
        #             math.radians(0.), [1, 0, 0]),
        #         tra.rotation_matrix(
        #             math.radians(0.0), [0, 1, 0]),
        #         tra.rotation_matrix(
        #             math.radians(0.0), [0, 0, 1]),
        #     ))

        self['edge_grasp']['ticket']['slide_transform_alt'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(0.), [1, 0, 0]),
                tra.rotation_matrix(
                    # math.radians(35.0), [0, 1, 0]),
                    math.radians(0.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 0, 1]),
            ))

        # for SH_V2
        # self['edge_grasp']['ticket']['pre_approach_transform_alt'] = tra.concatenate_matrices(
        #     tra.translation_matrix([0.055, -0.01, -0.10]),
        #     tra.concatenate_matrices(
        #         tra.rotation_matrix(
        #             math.radians(0.), [1, 0, 0]),
        #         tra.rotation_matrix(
        #             math.radians(-20.0), [0, 1, 0]),
        #         tra.rotation_matrix(
        #             math.radians(0.0), [0, 0, 1]),
        #     ))

        # this directly correlates with pre_approach_transform TODO: it shouldn't
        self['edge_grasp']['ticket']['palm_edge_offset_alt'] = -0.04

        # for SH_V2
        # self['edge_grasp']['ticket']['palm_edge_offset_alt'] = -0.06

        # TODO: preferably, we would be using this to set the ee-tf to an optimal pose for each respective object.
        # However, the sampling in the feasibility check module shifts the sampling manifold by the values set here,
        # instead of changing the kinematics, resulting in wrong samples.
        # see commit 0ef805d in tub_feasibility_check for further reference
        self['edge_grasp']['ticket']['hand_transform'] = tra.concatenate_matrices(
            tra.translation_matrix([0.0, 0.0, 0.0]),
            tra.concatenate_matrices(
                tra.rotation_matrix(
                    math.radians(90.0), [1, 0, 0]),
                tra.rotation_matrix(
                    math.radians(0.0), [0, 1, 0]),
                tra.rotation_matrix(
                    math.radians(90.0), [0, 0, 1]),
            ))

        # good values for Pisa/IIT Hand
        # self['edge_grasp']['ticket']['post_slide_pose_trajectory'] = np.array([
        #     tra.translation_matrix([0, 0, -0.045]),
        #     tra.concatenate_matrices(tra.rotation_matrix(math.radians(35.0), [0, 1, 0]),
        #                              tra.translation_matrix([0.04, 0.02, 0])),
        #     tra.translation_matrix([0, 0, 0.045]),
        #     tra.translation_matrix([-0.08, 0, 0])
        # ])

        self['edge_grasp']['ticket']['post_slide_pose_trajectory'] = np.array([
            tra.rotation_matrix(math.radians(35.0), [0, 1, 0]),
            tra.translation_matrix([0.00, 0, -0.035]),
            tra.translation_matrix([0.05, 0.01, 0]),
            # tra.rotation_matrix(math.radians(0.0), [0, 1, 0]),
            # tra.concatenate_matrices(tra.rotation_matrix(math.radians(35.0), [0, 1, 0]),
            #                          tra.translation_matrix([0.03, 0.00, -0.03])),
            tra.translation_matrix([0, 0, 0.035]),
            tra.translation_matrix([-0.07, 0, 0])
        ])

        # self['edge_grasp']['ticket']['post_slide_pose_trajectory'] = np.array([np.eye(4)])

        # # for SH_V2
        # self['edge_grasp']['ticket']['post_slide_pose_trajectory'] = np.array([
        #     tra.translation_matrix([0, 0, -0.035]),
        #     tra.concatenate_matrices(tra.rotation_matrix(math.radians(20.0), [0, 1, 0]),
        #                              tra.translation_matrix([0.02, 0.02, 0])),
        #     tra.translation_matrix([0, 0, 0.015]),
        #     tra.translation_matrix([-0.03, 0, -0.005]),
        #     tra.concatenate_matrices(tra.rotation_matrix(math.radians(10.0), [0, 1, 0]),
        #                              tra.translation_matrix([0, 0, -0.005])),
        # ])

        # 1: push
        # -1: pull
        self['edge_grasp']['ticket']['sliding_direction'] = -1

######################################################################


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

        # This defines the robot noise distribution for the grasp success estimator, as calculated by
        # calculate_success_estimator_object_params.py. First value is mean, second is standard deviation.
        # This is mainly robot specific, but depending on the accuracy of the hand models each hand might introduce
        # additional noise. In that case the values should be updated in their specific classes
        self['success_estimation_robot_noise'] = np.array([0.0323, 0.0151])

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

        self['edge_grasp']['object']['pre_grasp_manifold'] = Manifold(
            {'min_position_deltas': [-0.05, -0.05, -0.05],
             'max_position_deltas': [0.05, 0.05, 0.05],
             'min_orientation_deltas': [0, 0, 0],
             'max_orientation_deltas': [0, 0, 0],
             })

        self['edge_grasp']['object']['go_down_manifold'] = Manifold({'min_position_deltas': [-0.01, -0.04, -0.05],
                                                                        'max_position_deltas': [0.06, 0.04, 0.01],
                                                                        'min_orientation_deltas': [0, 0, 0],
                                                                        'max_orientation_deltas': [0, 0, 0]
                                                                        })

        self['edge_grasp']['object']['slide_to_edge_manifold'] = Manifold(
            {'min_position_deltas': [-0.04, -0.04, -0.01],
             'max_position_deltas': [0.04,  0.04,  0.01],
             'min_orientation_deltas': [0, 0, 0],
             'max_orientation_deltas': [0, 0, 0]
             })


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
