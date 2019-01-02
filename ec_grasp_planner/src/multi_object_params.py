#!/usr/bin/env python

import yaml
import math
import numpy as np
import tf
from tf import transformations as tra
from geometry_graph_msgs.msg import Node, geometry_msgs
import rospy
import tf_conversions.posemath as pm


import rospkg
from tornado.concurrent import return_future

USE_OCADO_HEURISTIC = True

if USE_OCADO_HEURISTIC:
    from target_selection_in_ifco import srv as target_selection_srv

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')

def unit_vector(vector):
    # Returns the unit vector of the vector.
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def convert_msg_to_homogenous_tf(msg):
    return pm.toMatrix(pm.fromMsg(msg))

def convert_homogenous_tf_to_msg(htf):
    return pm.toMsg(pm.fromMatrix(htf))

if USE_OCADO_HEURISTIC:
    def get_ec_index_in_node_list(heuristic_ec_index, grasp_type):
        # The q_mat return by the service call has always 5 columns
        if grasp_type == 'Any':
            return heuristic_ec_index
        elif grasp_type == 'WallGrasp':
            return heuristic_ec_index - 1
        elif grasp_type == 'SurfaceGrasp':
            return 0 

class multi_object_params:
    def __init__(self, file_name="object_param.yaml"):
        self.file_name = file_name
        self.data = None        

    def get_object_params(self):
        if self.data is None:
            self.load_object_params()
        return self.data
    ## --------------------------------------------------------- ##
    #load parameters for hand-object-strategy
    def load_object_params(self):
        file = pkg_path + '/data/' + self.file_name
        with open(file, 'r') as stream:
            try:
                self.data = yaml.load(stream)
                # print("data loaded {}".format(file))
            except yaml.YAMLError as exc:
                print(exc)

## --------------------------------------------------------- ##
    # return 0 or 1 if strategy is applicable on the object
    # if there is a list of possible outcomes thant the strategy is applicable
    def pdf_object_strategy(self, object):
        if isinstance(object['success'], list):
            return 1
        else:
            return object['success']

## --------------------------------------------------------- ##
    # return probability based on object and ec features
    def pdf_object_ec(self, object, ec_frame, strategy):
        q_val = -1
        success = object['success']
        object_frame = object['frame']

        # if object-ec angle is given, get h_val for this feature
        # h_angle(relative object orientation to EC):
        # the optimal orientation values +/- epsilon = x probability - given in the object_param.yaml
        if object.get('angle',0):
            obj_x_axis = object_frame[0:3, 0]

            for idx, val in enumerate(object['angle']):
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
        # the one on th right side of the robot
        # y coord is the smallest

        if all_ec_frames[current_ec_index][1,3] > 0:
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

    def black_list_unreachable_zones(self, object, object_params, ifco_in_base_transform, strategy):

        # this function will blacklist out of reach zones for wall and surface grasp
        if strategy not in ["WallGrasp", "SurfaceGrasp"]:
            return 1

        object_min = object_params['min']
        object_max = object_params['max']
        object_frame = object['frame']

        object_in_ifco_frame  = ifco_in_base_transform.dot(object_frame)

        if object_in_ifco_frame[0,3] > object_min[0]  \
            and object_in_ifco_frame[0,3] < object_max[0] \
            and object_in_ifco_frame[1,3] > object_min[1] \
            and object_in_ifco_frame[1,3] < object_max[1]:
            return 1
        else:
            return 0

## --------------------------------------------------------- ##
    # object-environment-hand based heuristic, q_value for grasping
    def heuristic(self, object, current_ec_index, strategy, all_ec_frames, ifco_in_base_transform):

        ec_frame = all_ec_frames[current_ec_index]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']
        q_val = 1
        q_val = q_val * \
                self.pdf_object_strategy(object_params) * \
                self.pdf_object_ec(object_params, ec_frame, strategy) * \
                self.black_list_unreachable_zones(object, object_params, ifco_in_base_transform, strategy)* \
                self.black_list_walls(current_ec_index, all_ec_frames, strategy)


        #print(" ** q_val = {} blaklisted={}".format(q_val, self.black_list_walls(current_ec_index, all_ec_frames)))
        return q_val

## --------------------------------------------------------- ##
    # find the max probability and if there are more than one return one randomly
    def argmax_h(self, Q_matrix):
        # find max probablity in list

        indeces_of_max = np.argwhere(Q_matrix == Q_matrix.max())
        print("indeces_of_max  = {}".format(indeces_of_max ))

        print Q_matrix
        if Q_matrix.max() == 0.0:
            rospy.logwarn("No Suitable Grasp Found - PLEASE REPLAN!!!")

        return indeces_of_max[0][0], indeces_of_max[0][1]

## --------------------------------------------------------- ##
    # samples from a pdf dictionary where the values are normalized
    # returns the key of the sample
    def sample_from_pdf(self, pdf_matrix):

        # reshape matrix to a vector for sampling
        pdf_array = np.reshape(pdf_matrix, pdf_matrix.shape[0]*pdf_matrix.shape[1] )

        #init vector for normalization
        pdf_normalized = np.zeros(len(pdf_array))

        # normalize pdf, if all 0 all are equally possible
        if sum(pdf_array) == 0:
            pdf_normalized[:] = 1.0/len(pdf_array)
        else:
            pdf_normalized = pdf_array/sum(pdf_array)

        # sample probabilistically
        sampled_item = (np.random.choice(len(pdf_normalized), p=pdf_normalized))

        return sampled_item // pdf_matrix.shape[1], sampled_item % pdf_matrix.shape[1]

## --------------------------------------------------------- ##
    # chose random object and ec
    # the ec shoudl be valied for the given object
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


## --------------------------------------------------------- ##
    # function called to process all objects and ECs
    # assumption1: all objects are the same type
    # objects is a dictionary with obilagorty keys: type, frame (in robot base frame)
    # ecs is a list of graph nodes (see geometry_graph)
    def process_objects_ecs(self, objects, ecs, graph_in_base, ifco_in_base_transform,  SG_pre_grasp_in_object_frame, WG_pre_grasp_in_object_frame, h_process_type = "Deterministic", grasp_type = "Any", object_list_msg = []):

        # print("object: {}, \n ecs: {} \n graphTF: {}, h_process: {}".format(objects, ecs, graph_in_base, h_process_type))
        # print("ec type: {}".format(type(ecs[0])))
        # load parameter file
        self.load_object_params()       

        if USE_OCADO_HEURISTIC:
            srv = rospy.ServiceProxy('generate_q_matrix', target_selection_srv.GenerateQmatrix)
            graspable_with_any_hand_orientation = False
            SG_success_rate = 1.0
            WG_success_rate = 1.0

            # currently camera_in_base = graph_in_base
            camera_in_ifco = np.inv(ifco_in_base_transform).dot(graph_in_base) 
            camera_in_ifco_msg = convert_homogenous_tf_to_msg(camera_in_ifco)

            SG_pre_grasp_in_object_frame_msg = convert_homogenous_tf_to_msg(SG_pre_grasp_in_object_frame)
            WG_pre_grasp_in_object_frame_msg = convert_homogenous_tf_to_msg(WG_pre_grasp_in_object_frame)
            ifco_in_base_msg = convert_homogenous_tf_to_msg(ifco_in_base_transform)

            res = srv(grasp_type, object_list_msg, camera_in_ifco_msg, SG_pre_grasp_in_object_frame_msg, WG_pre_grasp_in_object_frame_msg, ifco_in_base_msg, graspable_with_any_hand_orientation, SG_success_rate, WG_success_rate)
            Q_list = res.Q_mat.data
            number_of_columns = 5 #the service always returns a matrix with 5 ecs, but compute heuristic for the desired grasp only (0 otherwise)
            Q_matrix = np.matrix(Q_list).reshape((len(objects), number_of_columns))

        else:
            Q_matrix = np.zeros((len(objects),len(ecs)))
            # iterate through all objects
            for i,o in enumerate(objects):

                # check if the given hand type for this object is set in the yaml
                # print ("object type: {}".format(o["type"]))

                if not self.data[o["type"]]:
                    print("The given object {} has no parameters set in the yaml {}".format(o["type"], self.file_name))
                    return -1

                all_ec_frames = []
                for j, ec in enumerate(ecs):
                    all_ec_frames.append(graph_in_base.dot(convert_msg_to_homogenous_tf(ec.transform)))
                    print("ecs:{}".format(graph_in_base.dot(convert_msg_to_homogenous_tf(ec.transform))))

                for j,ec in enumerate(ecs):
                    # the ec frame must be in the same reference frame as the object
                    ec_frame_in_base = graph_in_base.dot(convert_msg_to_homogenous_tf(ec.transform))
                    Q_matrix[i,j] = self.heuristic(o, j, ec.label, all_ec_frames, ifco_in_base_transform)

        # print (" ** h_mx = {}".format(Q_matrix))

        # select heuristic function for choosing object and EC
        #argmax samples from the [max (H(obj, ec)] list
        if h_process_type == "Deterministic":
            object_index,  ec_index = self.argmax_h(Q_matrix)
            print(" ** h_mx[{}, {}]".format(object_index, ec_index))
            print(" ** h_mx[{}, {}]".format(object_index, ecs[ec_index]))            
        # samples from [H(obj, ec)] list
        elif h_process_type == "Probabilistic":
            object_index, ec_index = self.sample_from_pdf(Q_matrix)            
        elif h_process_type == "Random":
            object_index, ec_index = self.random_from_Qmatrix(Q_matrix)        
        
        ## Compute pre_grasp_pose
        if USE_OCADO_HEURISTIC:
            srv = rospy.ServiceProxy('get_pregrasp_pose_q_row_col', target_selection_srv.GetPreGraspPoseForQRowCol)
            res = srv(object_index, ec_index)
            pre_grasp_pose_in_base_frame = convert_msg_to_homogenous_tf(res.pre_grasp_pose_in_base_frame)
            ec_index = get_ec_index_in_node_list(ec_index, grasp_type)        
        else:
            object_pose = objects[object_index]['frame']
            chosen_node = ecs[ec_index]            
            if chosen_node.label == 'SurfaceGrasp':
                pre_grasp_pose_in_base_frame = object_pose.dot(SG_pre_grasp_in_object_frame)
            elif chosen_node.label == 'WallGrasp':
                object_pos_with_ec_orientation = graph_in_base.dot(convert_msg_to_homogenous_tf(chosen_node.transform))
                object_pos_with_ec_orientation[:3,3] = tra.translation_from_matrix(object_pose)
                pre_grasp_pose_in_base_frame = object_pos_with_ec_orientation.dot(WG_pre_grasp_in_object_frame)
            else:
                raise ValueError("Unknown grasp type: {}".format(chosen_node.label))

        return object_index, ec_index, pre_grasp_pose_in_base_frame


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
