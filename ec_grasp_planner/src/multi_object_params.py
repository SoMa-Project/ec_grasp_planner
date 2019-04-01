#!/usr/bin/env python

import yaml
import math
import numpy as np
from tf import transformations as tra
from geometry_graph_msgs.msg import Node, geometry_msgs, ObjectList
import rospy
import tf_conversions.posemath as pm
import random

import rospkg
import pkgutil
from tornado.concurrent import return_future

from functools import partial
from planner_utils import *

import tub_feasibility_check_interface
import GraspFrameRecipes

# Check if ocado reachability service is installed and load it, if it is
ocado_reachability_loader = pkgutil.find_loader('target_selection_in_ifco')
if ocado_reachability_loader is not None:
    try:
        from target_selection_in_ifco import srv as target_selection_srv
    except ImportError:
        rospy.logwarn('The Ocado target selection module was not loaded. This means it cannot be used')

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')


# CODE DUPLICATION... 
# TODO: DEAL WITH THIS
# ================================================================================================
def get_derived_corner_grasp_frames(corner_frame, object_pose):

    ec_frame = np.copy(corner_frame)
    ec_frame[:3, 3] = tra.translation_from_matrix(object_pose)
    # y-axis stays the same, lets norm it just to go sure
    y = ec_frame[:3, 1] / np.linalg.norm(ec_frame[:3, 1])
    # z-axis is (roughly) the vector from corner to object
    z = ec_frame[:3, 3] - corner_frame[:3, 3]
    # z-axis should lie in the y-plane, so we subtract the part that is perpendicular to the y-plane
    z = z - (np.dot(z, y) * y)
    z = z / np.linalg.norm(z)
    # x-axis is perpendicular to y- and z-axis, again normed to go sure
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    # the rotation part is overwritten with the new axis
    ec_frame[:3, :3] = tra.inverse_matrix(np.vstack((x, y, z)))

    corner_frame_alpha_zero = np.copy(corner_frame)
    corner_frame_alpha_zero[:3, :3] = np.copy(ec_frame[:3, :3])

    return ec_frame, corner_frame_alpha_zero



class multi_object_params:
    def __init__(self, file_name="object_param.yaml"):
        self.file_name = file_name
        self.data = None
        self.stored_trajectories = {}

        self.hand_name = rospy.get_param('/planner_gui/hand', default='RBOHandP24_pulpy')
        self.heuristic_type = rospy.get_param("planner_gui/heuristic_type", default="tub-separated")

    def get_object_params(self):
        if self.data is None:
            self.load_object_params()
        return self.data

    # This function will return a dictionary, mapping a motion name (e.g. pre_grasp) to an alternative behavior
    # (e.g. a sequence of joint states) to the default hard-coded motion in the planner.py (in case the tub feasibility
    # checker decided that it was necessary to generate).
    # If for the given object-ec-pair no such alternative behavior was created, this function returns None.
    def get_alternative_behavior(self, object_idx, ec_index):
        print(self.stored_trajectories)
        if (object_idx, ec_index) not in self.stored_trajectories:
            return None
        return self.stored_trajectories[(object_idx, ec_index)]

    # This function deletes all stored trajectories (aka. alternative behavior)
    def reset_kinematic_checks_information(self):
        self.stored_trajectories = {}

    # --------------------------------------------------------- #
    # load parameters for hand-object-strategy
    def load_object_params(self):
        file = pkg_path + '/data/' + self.file_name
        with open(file, 'r') as stream:
            try:
                self.data = yaml.load(stream)
                # print("data loaded {}".format(file))
            except yaml.YAMLError as exc:
                print(exc)

    # --------------------------------------------------------- #
    # return 0 or 1 if strategy is applicable on the object
    # if there is a list of possible outcomes then the strategy is applicable
    def pdf_object_strategy(self, object):
        if isinstance(object['success'][self.hand_name], list):
            return 1
        else:
            return 1 if object['success'][self.hand_name] > 0 else 0

    # --------------------------------------------------------- #
    # return probability based on object and ec features
    def pdf_object_ec(self, object, ec_frame, strategy):
        q_val = -1
        success = object['success'][self.hand_name]
        object_frame = object['frame']

        # if object-ec angle is given, get h_val for this feature
        # h_angle(relative object orientation to EC):
        # the optimal orientation values +/- epsilon = x probability - given in the object_param.yaml
        if isinstance(success, list):
            obj_x_axis = object_frame[0:3, 0]

            for idx, val in enumerate(object['angle'][self.hand_name]):
                ec_x_axis = ec_frame[0:3, 0]
                angle_epsilon = object['epsilon']
                diff_angle = math.fabs(angle_between(obj_x_axis, ec_x_axis) - math.radians(val))
                # print("obj_x = {}, ec_x = {}, eps = {}, optimalDeg = {}, copare = {}".format(
                #     obj_x_axis, ec_x_axis, angle_epsilon, val, diff_angle))
                if diff_angle <= math.radians(angle_epsilon):
                    q_val = success[idx]
                    break
            # if the angle was not within the given bounded sets
            # take the last value from the list of success values
            if q_val == -1:
                q_val = success[-1]
                # print (" *** no good angle found")
            # if there are no other criteria for q_val
        else:
            q_val = success

        # distance form EC (wall end edge)
        # this is the tr from object_frame to ec_frame in object frame
        if strategy in ["WallGrasp", "EdgeGrasp"]:
            delta = np.linalg.inv(ec_frame).dot(object_frame)
            # this is the distance between object and EC
            dist = delta[2, 3]
            # include distance to q_val, longer distance decreases q_val
            q_val = q_val * (1/dist)

        return q_val

    def black_list_walls(self, current_ec_index, all_ec_frames, strategy):

        if strategy not in ["WallGrasp", "EdgeGrasp"]:
            return 1
        # this function will blacklist all walls except
        # the one on the right side of the robot
        # y coord is the smallest

        if all_ec_frames[current_ec_index][1, 3] > 0:
                return 0

        min_y = 10000
        min_y_index = 0

        for i, ec in enumerate(all_ec_frames):
            if min_y > ec[1,3]:
                min_y = ec[1,3]
                min_y_index = i

        if min_y_index == current_ec_index:
            return 1
        else:
            return 0

    def black_list_corners(self, current_ec_index, all_ec_frames, strategy):

        if strategy not in ["CornerGrasp"]:
            # print("strategy {} not CG".format(strategy))
            return 1
        # this function will blacklist all corners except
        # the one on th right side and further from the robot
        # y coord is negative and x is greater

        # print("Corner EC: \n {} \n at x: {} \n at y: {}".format(all_ec_frames[current_ec_index],all_ec_frames[current_ec_index][0,3], all_ec_frames[current_ec_index][1,3]))

        if all_ec_frames[current_ec_index][1,3] > 0:
            # print("==== F1 ========")
            return 0

        max_x = -10000.0
        max_x_index = 0

        for i, ec in enumerate(all_ec_frames):
            if  (max_x < ec[0,3]) and ec[1,3] < 0:
                max_x = ec[0,3]
                max_x_index = i

        if max_x_index == current_ec_index:
            # print("==== true ========")
            return 1
        else:
            # print("==== F2 i={} c={} ========".format(max_x_index, current_ec_index))
            return 0

    def black_list_unreachable_zones(self, object, object_params, ifco_in_base_transform, strategy):

        # this function will blacklist out of reach zones for wall and surface grasp
        if strategy not in ["WallGrasp", "SurfaceGrasp"]:
            return 1

        object_min = object_params['min']
        object_max = object_params['max']
        object_frame = object['frame']

        object_in_ifco_frame = tra.inverse_matrix(ifco_in_base_transform).dot(object_frame)

        if object_in_ifco_frame[0,3] > object_min[0]  \
            and object_in_ifco_frame[0,3] < object_max[0] \
            and object_in_ifco_frame[1,3] > object_min[1] \
            and object_in_ifco_frame[1,3] < object_max[1]:
            return 1
        else:
            return 0

    def surfaceGrasp_hihgest_object(self, object, objects, strategy):
        # for surface grasp always prefer the hiehest object on the pile
        if strategy not in ["SurfaceGrasp"]:
            print("Not SG!")
            return 1

        object_HT = object['frame']
        max_h = -10000
        epsilon = 0.01

        for o in objects:
            o_HT = o['frame']
            print("o_h: {}".format(o_HT[2,3]))
            if o_HT[2,3] > max_h:
                max_h = o_HT[2,3]
        print("o_max: {}".format(max_h))

        if object_HT[2,3]+epsilon < max_h:
            return 0.2

        return 1.0

    def wallGrasp_distant_object(self, object, objects, strategy, ec_HT):
        # for surface grasp always prefer the hiehest object on the pile
        if strategy not in ["WallGrasp", "CornerGrasp"]:
            return 1

        object_HT = object['frame']
        max_dist = -10000
        epsilon = 0.01

        for o in objects:
            o_HT = o['frame']

            delta = np.linalg.inv(ec_HT).dot(o_HT)
            # this is the distance between object and EC
            dist = delta[2, 3]

            if dist > max_dist:
                max_dist = dist

        delta = np.linalg.inv(ec_HT).dot(object_HT)
        # this is the distance between object and EC
        dist = delta[2, 3]

        if dist+epsilon < max_dist:
            return 0.2

        return 1.0

    def cornerGrasp_distant_object(self, object, objects, strategy, ec_HT):
        # for surface grasp always prefer the hiehest object on the pile
        if strategy not in ["CornerGrasp"]:
            return 1

        object_HT = object['frame']
        max_dist = -10000
        epsilon = 0.01

        #object_in_EC = np.linalg.inv(ec_HT)*object_HT
        # angle_of_attack = np.arctan2(object_HT[0,3], object_HT[2,3])

        ##z = z - (np.dot(z, y) * y) project to surface
        attack_vector =  object_HT[0:2,3] - ec_HT[0:2,3]
        attack_vector =  attack_vector - (np.dot(attack_vector, ec_HT[0:2,1]) *ec_HT[0:2,1] )

        angle_of_attack = np.arccos(np.dot(ec_HT[0:2,2], attack_vector)/
                                    (np.linalg.norm(ec_HT[0:2,2]) * np.linalg.norm(attack_vector)))

        # print("angle: {} frame {}, obj {}".format(np.rad2deg(angle_of_attack), ec_HT[0:3,3], object_HT[0:3,3]))

        if angle_of_attack > np.deg2rad(30) or angle_of_attack < np.deg2rad(20):
            return 0.2

        # for o in objects:
        #     o_HT = o['frame']
        #
        #     attack_vector = o_HT[0:2, 3] - ec_HT[0:2, 3]
        #     attack_vector = attack_vector - (np.dot(attack_vector, ec_HT[0:2, 1]) * ec_HT[0:2, 1])
        #     angle_of_attack = np.arccos(np.dot(ec_HT[0:2, 2], attack_vector) / \
        #                                 (np.linalg.norm(ec_HT[0:2, 2]) * np.linalg.norm(attack_vector)))
        #
        #     if angle_of_attack < np.deg2rad(-35) and angle_of_attack > np.deg2rad(-18):
        #         delta = np.linalg.inv(ec_HT).dot(o_HT)
        #         # this is the distance between object and EC
        #         dist = delta[2, 3]
        #
        #         if dist > max_dist:
        #             max_dist = dist
        #
        # delta = np.linalg.inv(ec_HT).dot(object_HT)
        # # this is the distance between object and EC
        # dist = delta[2, 3]
        #
        # if dist+epsilon < max_dist:
        #     return 0.2

        return 1.0

    def black_list_risk_regions(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames,
                                ifco_in_base_transform):

        object = objects[current_object_idx]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        zone_fac = self.black_list_unreachable_zones(object, object_params, ifco_in_base_transform, strategy)
        wall_fac = self.black_list_walls(current_ec_index, all_ec_frames, strategy)
        corner_fac = self.black_list_corners(current_ec_index, all_ec_frames, strategy)

        return zone_fac * wall_fac * corner_fac

    def basic_pile_heuristic(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames):
        ##-----------------------------------OCE-------------------
        ## used for ECE and OCE
        # self.black_list_walls(current_ec_index, all_ec_frames, strategy) * \
        ## used for ECE and OCE
        # self.black_list_corners(current_ec_index, all_ec_frames, strategy) * \
        ## used for OCE only @ corner - can be active for all OCE strategies
        #self.cornerGrasp_distant_object(object, objects, strategy, ec_frame) * \
        ## used for OCE only @ corenr & wall - can be active for all OCE strategies
        #self.wallGrasp_distant_object(object, objects, strategy, ec_frame) * \
        ## used for OCE only @ surface - can be active for all OCE strategies
        #self.surfaceGrasp_hihgest_object(object, objects, strategy)

        object = objects[current_object_idx]
        ec_frame = all_ec_frames[current_ec_index]

        return self.black_list_walls(current_ec_index, all_ec_frames, strategy) * \
            self.black_list_corners(current_ec_index, all_ec_frames, strategy) * \
            self.cornerGrasp_distant_object(object, objects, strategy, ec_frame) * \
            self.wallGrasp_distant_object(object, objects, strategy, ec_frame) * \
            self.surfaceGrasp_hihgest_object(object, objects, strategy)


    # ------------------------------------------------------------- #
    # object-environment-hand based heuristic, q_value for grasping
    def heuristic(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames,
                      ifco_in_base_transform, handarm_params):

        object = objects[current_object_idx]

        ec_frame = all_ec_frames[current_ec_index]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        if self.heuristic_type == 'tub-separated':
            feasibility_fun = partial(tub_feasibility_check_interface.check_kinematic_feasibility,
                                      current_object_idx, objects, object_params, current_ec_index,
                                      strategy, all_ec_frames, ifco_in_base_transform, handarm_params,
                                      self.stored_trajectories)

        elif self.heuristic_type == 'tub-pile':
            # TODO integrate
            raise ValueError("Not supported yet")

        elif self.heuristic_type == 'basic-separated':
            # Use default plain and simple black listing approach
            feasibility_fun = partial(self.black_list_risk_regions, current_object_idx, objects, current_ec_index,
                                      strategy, all_ec_frames, ifco_in_base_transform)

        elif self.heuristic_type == 'basic-pile':
            # Use default plain and simple black listing approach
            feasibility_fun = partial(self.basic_pile_heuristic, current_object_idx, objects, current_ec_index,
                                      strategy, all_ec_frames)

        else:
            if self.heuristic_type == 'ocado':
                raise ValueError("Fatal: This function should never be called with ocado reachability checker!")
            raise ValueError("The heuristic type {} is not supported yet".format(self.heuristic_type))

        q_val = 1
        q_val = q_val * \
            self.pdf_object_strategy(object_params) * \
            self.pdf_object_ec(object_params, ec_frame, strategy) * \
            feasibility_fun()

        # print(" ** q_val = {} blaklisted={}".format(q_val, self.black_list_walls(current_ec_index, all_ec_frames)))
        return q_val

    # --------------------------------------------------------- #
    # find the max probability and if there are more than one return one randomly
    def argmax_h(self, Q_matrix):
        # find max probablity in list        

        indeces_of_max = np.argwhere(Q_matrix == Q_matrix.max())
        # print("indeces_of_max  = {}".format(indeces_of_max ))

        # print Q_matrix
        if Q_matrix.max() == 0.0:
            rospy.logwarn("No Suitable Grasp Found - PLEASE REPLAN!!!")

        if len(indeces_of_max) > 1:
            # if several max element, pick one randomly
            max_ind = random.SystemRandom().choice(indeces_of_max)
        else:
            max_ind = indeces_of_max[0]

        return max_ind[0], max_ind[1]

    # --------------------------------------------------------- #
    # samples from a pdf dictionary where the values are normalized
    # returns the key of the sample
    def sample_from_pdf(self, pdf_matrix):

        # reshape matrix to a vector for sampling
        pdf_array = np.reshape(pdf_matrix, pdf_matrix.shape[0]*pdf_matrix.shape[1] )

        # init vector for normalization
        pdf_normalized = np.zeros(len(pdf_array))

        # normalize pdf, if all 0 all are equally possible
        if sum(pdf_array) == 0:
            pdf_normalized[:] = 1.0/len(pdf_array)
        else:
            pdf_normalized = pdf_array/sum(pdf_array)

        # sample probabilistically
        sampled_item = (np.random.choice(len(pdf_normalized), p=pdf_normalized))

        return sampled_item // pdf_matrix.shape[1], sampled_item % pdf_matrix.shape[1]

    # --------------------------------------------------------- #
    # chose random object and ec
    # the ec should be valid for the given object
    # if there are no valid EC then pick randomly from all
    def random_from_Qmatrix(self, pdf_matrix):

        # reshape matrix to a vector for sampling
        pdf_array = np.reshape(pdf_matrix, pdf_matrix.shape[0] * pdf_matrix.shape[1])

        # init vector for normalization
        pdf_normalized = np.zeros(len(pdf_array))

        # normalize pdf, if all are equally possible
        pdf_normalized[:] = 1.0 / len(pdf_array)

        # sample probabilistically
        sampled_item = (np.random.choice(len(pdf_normalized), p=pdf_normalized))

        return sampled_item // pdf_matrix.shape[1], sampled_item % pdf_matrix.shape[1]

    def object_from_object_list_msg(self, object_type, object_list_msg, camera_in_base):
        obj_tmp = {} # object is a dictionary with obilagorty keys: type, frame (in robot base frame)
        obj_tmp['type'] = object_type
        # the TF must be in the same reference frame as the EC frames
        # Get the object frame in robot base frame
        object_in_camera = convert_pose_msg_to_homogeneous_tf(object_list_msg.transform.pose)
        object_in_base = camera_in_base.dot(object_in_camera)
        obj_tmp['frame'] = object_in_base
        obj_tmp['bounding_box'] = object_list_msg.boundingbox
        return obj_tmp

    def create_q_matrix(self, object_type, ecs, graph_in_base, ifco_in_base_transform,
                        SG_pre_grasp_in_object_frame=tra.identity_matrix(),
                        WG_pre_grasp_in_object_frame=tra.identity_matrix(),
                        grasp_type="Any", object_list_msg=ObjectList(), handarm_parameters=None):

        if self.heuristic_type == 'ocado':
            srv = rospy.ServiceProxy('generate_q_matrix', target_selection_srv.GenerateQmatrix)
            object_data = self.data[
                object_type]  # using the first object, in theory in the ocado use case we work with the same objects
            SG_success_rate = object_data['SurfaceGrasp']['success'][self.hand_name]
            WG_success_rate = object_data['WallGrasp']['success'][self.hand_name]
            CG_success_rate = object_data['CornerGrasp']['success'][self.hand_name]
            graspable_with_any_hand_orientation = object_data['graspable_with_any_hand_orientation']

            # currently camera_in_base = graph_in_base
            camera_in_ifco = np.linalg.inv(ifco_in_base_transform).dot(graph_in_base)
            camera_in_ifco_msg = convert_homogeneous_tf_to_pose_msg(camera_in_ifco)

            SG_pre_grasp_in_object_frame_msg = convert_homogeneous_tf_to_pose_msg(SG_pre_grasp_in_object_frame)
            WG_pre_grasp_in_object_frame_msg = convert_homogeneous_tf_to_pose_msg(WG_pre_grasp_in_object_frame)
            ifco_in_base_msg = convert_homogeneous_tf_to_pose_msg(ifco_in_base_transform)

            res = srv(grasp_type, object_list_msg, camera_in_ifco_msg, SG_pre_grasp_in_object_frame_msg,
                      WG_pre_grasp_in_object_frame_msg, ifco_in_base_msg, graspable_with_any_hand_orientation,
                      SG_success_rate, WG_success_rate)
            Q_list = res.Q_mat.data
            number_of_columns = len(ecs)
            Q_matrix = np.array(Q_list).reshape((len(object_list_msg.objects), number_of_columns))

        else:
            # build list of objects
            objects = [self.object_from_object_list_msg(object_type, o, graph_in_base) for o in object_list_msg.objects]

            Q_matrix = np.zeros((len(objects), len(ecs)))

            # iterate through all objects
            for i, o in enumerate(objects):

                # check if the given hand type for this object is set in the yaml
                # print ("object type: {}".format(o["type"]))

                if not self.data[o["type"]]:
                    rospy.logerr("The given object {} has no parameters set in the yaml {}".format(o["type"],
                                                                                                   self.file_name))
                    return None

                all_ec_frames = []
                for j, ec in enumerate(ecs):
                    all_ec_frames.append(graph_in_base.dot(convert_transform_msg_to_homogeneous_tf(ec.transform)))
                    print("ecs:{}".format(graph_in_base.dot(convert_transform_msg_to_homogeneous_tf(ec.transform))))

                for j, ec in enumerate(ecs):
                    # the ec frame must be in the same reference frame as the object
                    ec_frame_in_base = graph_in_base.dot(convert_transform_msg_to_homogeneous_tf(ec.transform))
                    Q_matrix[i, j] = self.heuristic(i, objects, j, ec.label, all_ec_frames,
                                                    ifco_in_base_transform, handarm_parameters)

        return Q_matrix

    # --------------------------------------------------------- #
    # function called to process all objects and ECs
    # assumption1: all objects are the same type
    # parameters:
    #   object_type currently set in the gui
    #   ecs is a list of graph nodes (see geometry_graph)
    #   graph_in_base transform of the camera in the robot base
    #   ifco_in_base_transform
    #   SG_pre_grasp_in_object_frame the initial pre-grasp pose in the object frame for SurfaceGrasp
    #   WG_pre_grasp_in_object_frame the initial pre-grasp pose in the object frame for WallGrasp towards the first wall
    #   CG_pre_grasp_in_object_frame the initial pre-grasp pose in the object frame for CornerGrasp
    #   h_process_type = ['Random', 'Deterministic', 'Probabilistic'] set in the gui
    #   grasp_type = ['Any', 'SurfaceGrasp', 'WallGrasp'] set in the gui
    #   object_list_msg is the object list returned by the vision service node (see geometry_graph)
    #   handarm_params are the hand-arm-specific parameters (see: handarm_parameters.py)
    # returns:    
    #   chosen_object (dict with key 'frame', 'bounding_box', 'type', 'index')
    #   chosen_node the chosen ec node from the geometry graph
    #   pre_grasp_pose_in_base_frame the initial pre-grasp pose in the robot frame for the chosen grasp
    def process_objects_ecs(self, object_type, ecs, graph_in_base, ifco_in_base_transform,
                            SG_pre_grasp_in_object_frame=tra.identity_matrix(),
                            WG_pre_grasp_in_object_frame=tra.identity_matrix(),
                            CG_pre_grasp_in_object_frame=tra.identity_matrix(),
                            h_process_type="Deterministic", grasp_type="Any", object_list_msg=ObjectList(),
                            handarm_parameters=None):

        # print("object: {}, \n ecs: {} \n graphTF: {}, h_process: {}".format(objects, ecs, graph_in_base, h_process_type))
        # print("ec type: {}".format(type(ecs[0])))
        # load parameter file
        self.load_object_params()
        self.reset_kinematic_checks_information()
        self.hand_name = rospy.get_param('/planner_gui/hand', default='RBOHandP24_pulpy')
        # set the heuristic type that should be used. Make sure this object member is only set here!
        self.heuristic_type = rospy.get_param("planner_gui/heuristic_type", default="tub-separated")

        # Calculate Q-Matrix
        Q_matrix = self.create_q_matrix(object_type, ecs, graph_in_base, ifco_in_base_transform,
                                        SG_pre_grasp_in_object_frame, WG_pre_grasp_in_object_frame, grasp_type,
                                        object_list_msg, handarm_parameters)

        print("Qmat = {}".format(Q_matrix))

        # Check if there is a grasp candidate
        if Q_matrix.max() == 0.0:
            rospy.logwarn("No suitable Grasp Found! Qmat = {}".format(Q_matrix))
            return None, None, None

        # select heuristic function for choosing object and EC
        if h_process_type == "Deterministic":
            # argmax samples from the [max (H(obj, ec)] list
            object_index,  ec_index = self.argmax_h(Q_matrix)
            print(" ** h_mx[{}, {}]".format(object_index, ec_index))
            print(" ** h_mx[{}, {}]".format(object_index, ecs[ec_index]))

        elif h_process_type == "Probabilistic":
            # samples from [H(obj, ec)] list
            object_index, ec_index = self.sample_from_pdf(Q_matrix)

        elif h_process_type == "Random":
            object_index, ec_index = self.random_from_Qmatrix(Q_matrix)

        else:
            raise ValueError("Unknown heuristic function for choosing object-ec pair: {}".format(h_process_type))

        # Compute pre_grasp_pose
        if self.heuristic_type == 'ocado':
            chosen_object = self.object_from_object_list_msg(object_type, object_list_msg.objects[object_index], graph_in_base)
            srv = rospy.ServiceProxy('get_pregrasp_pose_q_row_col', target_selection_srv.GetPreGraspPoseForQRowCol)            
            res = srv(object_index, ec_index)
            pre_grasp_pose_in_base_frame = convert_pose_msg_to_homogeneous_tf(res.pre_grasp_pose_in_base_frame)
        else:
            objects = [self.object_from_object_list_msg(object_type, o, graph_in_base) for o in object_list_msg.objects]
            chosen_object = objects[object_index]
            object_pose = chosen_object['frame']
            chosen_node = ecs[ec_index]

            if chosen_node.label == 'SurfaceGrasp':
                pre_grasp_pose_in_base_frame = GraspFrameRecipes.get_surface_pregrasp_pose_in_base_frame(
                    SG_pre_grasp_in_object_frame, object_pose)

            elif chosen_node.label == 'WallGrasp':
                pre_grasp_pose_in_base_frame = GraspFrameRecipes.get_wall_pregrasp_pose_in_base_frame(
                    chosen_node, WG_pre_grasp_in_object_frame, object_pose, graph_in_base)

            elif chosen_node.label == 'CornerGrasp':
                pre_grasp_pose_in_base_frame = GraspFrameRecipes.get_corner_pregrasp_pose_in_base_frame(
                    chosen_node, CG_pre_grasp_in_object_frame, object_pose, graph_in_base)

            else:
                raise ValueError("Unknown grasp type: {}".format(chosen_node.label))       
        
        chosen_object['index'] = object_index

        return chosen_object, ec_index, pre_grasp_pose_in_base_frame


def test(ece_list = []):
    # this is only a test code to show usability of the library
    if len(ece_list) == 0:
        return "init ece list with nodes form the ECE graph"

    # object has a frame, type (see use-case types and input PDF function), and bounding box properties
    object =  {'frame': np.array([[-0.99997823, -0.00579027, -0.00319919,  0.54917589],
       [ 0.0057939 , -0.99998269, -0.0011255 , -0.00102592],
       [-0.00319261, -0.00114401,  0.99999436,  0.35815563],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                'type': "punnet",
                'bounding_box': {'x': 0.118688985705, 'y': 0.0980169996619, 'z': 0.0797315835953}}

    # list of objects
    objects = [object]

    # an EC is an  the ECE_Graph node: transformation and a label

    # ec1 = Node
    # ec1.label = "WallGrasp"
    # ec1.transform = geometry_msgs.msg.TransformStamped
    # ec1.transform = tra.concatenate_matrices(tra.translation_matrix([-0.532513504798, 0.222529488642, 1.39476392907]),
    #                                          tra.rotation_matrix(math.radians(-70.0), [0, 0, 1]))
    #
    # # tra.concatenate_matrices(tra.translation_matrix([-0.532513504798, 0.222529488642, 1.39476392907]), tra.rotation_matrix(math.radians(170.0), [0, 0, 1]))
    #
    # ec2 = Node
    # ec2.label = "WallGrasp"
    # ec2.transform = tra.concatenate_matrices(tra.translation_matrix([-0.532513504798, 0.222529488642, 1.39476392907]),
    #                                          tra.rotation_matrix(math.radians(-70.0), [0, 0, 1]))



    # list of all available ECs
    list_of_eces = ece_list

    # this is a transformation that brings the ec frames in the same refernece frame as for the objects
    graphTransform = np.array([[4.79425539e-01, - 6.02725216e-01,   6.37866340e-01, 0.00000000e+00],
              [-8.77582562e-01, - 3.29270286e-01,   3.48467970e-01, - 7.00000000e-01],
    [3.50502960e-12, - 7.26844821e-01, - 6.86801723e-01, 1.40000000e+00],
    [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    # heuristic function can be Random, Deterministic, or Probabilistic
    heuristic_function = "Deterministic"

    # init object to process multi objects
    foo = multi_object_params()
    # load object and ec related probability distribution function
    foo.load_object_params()
    # find object-ec tuple based on the selected heuristic function
    obj_chosen_idx, ec_chosen_idx = foo.process_objects_ecs(objects, list_of_eces, graphTransform, heuristic_function)

    obj_chosen = objects[obj_chosen_idx]
    ec_chosen = list_of_eces[ec_chosen_idx]

    print("Chosen object = {} \n\n Exploiting ec = {}".format(obj_chosen, ec_chosen))

    # h_val = foo.heuristic(obj, ec, strategy, hand)
    #
    #
    # print("H({}, {}, {}) = {}".format(obj["type"], strategy, hand, h_val))
