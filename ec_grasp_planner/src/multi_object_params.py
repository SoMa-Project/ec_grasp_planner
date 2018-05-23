#!/usr/bin/env python

import yaml
# import rospy
import math
import numpy as np
import tf
from tf import transformations as tra
import operator
import random

import rospkg
from tornado.concurrent import return_future

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

class multi_object_params:
    def __init__(self, file_name = "object_param.yaml"):
        self.file_name = file_name

    # ================================================================================================
    def transform_msg_to_homogenous_tf(self, msg):
        return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]),
                      tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))

    ## --------------------------------------------------------- ##
    #load parameters for hand-object-strategy
    def load_object_params(self):
        file = pkg_path + '/data/'+ self.file_name
        with open(file, 'r') as stream:
            try:
                self.data =yaml.load(stream)
                # print("data loaded")
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
                if (diff_angle <= math.radians(angle_epsilon)):
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
        if (strategy in ["WallGrasp", "EdgeGrasp"]):
            delta = np.linalg.inv(ec_frame).dot(object_frame)
            # this is the distance between object and EC
            dist = delta[2, 3]
            # include distance to q_val, longer distance decreases q_val
            q_val = q_val * (1/dist)

        return q_val

## --------------------------------------------------------- ##
    # object-environment-hand based heuristic, q_value for grasping
    def heuristic(self, object, ec_frame, strategy):

        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']
        q_val = 1
        q_val = q_val * self.pdf_object_strategy(object_params) * self.pdf_object_ec(object_params, ec_frame, strategy)
        # print(" ** q_val = {}".format(q_val))
        return q_val

## --------------------------------------------------------- ##
    # find the max probability and if there are more than one return one randomly
    def argmax_h(self, Q_matrix):
        # find max probablity in list

        ideces_of_max = np.argwhere(Q_matrix == Q_matrix.max())
        print("ideces_of_max  = {}".format(ideces_of_max ))

        return ideces_of_max[0][0], ideces_of_max[0][1]

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
    def process_objects_ecs(self, objects, ecs, graph_in_base, h_process_type="argmax"):

        # load parameter file
        self.load_object_params()

        Q_matrix = np.zeros((len(objects),len(ecs)))

        # iterate through all objects
        for i,o in enumerate(objects):

            # check if the given hand type for this object is set in the yaml
            if not self.data[o["type"]]:
                print("The given object {} has no parameters set in the yaml {}".format(o["type"], self.file_name))
                return -1

            for j,ec in enumerate(ecs):
                # the ec frame must be in the same reference frame as the object
                ec_frame_in_base = graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform))
                Q_matrix[i,j] = self.heuristic(o, ec_frame_in_base, ec.label)

        print (" ** h_mx = {}".format(Q_matrix))

        # select heuristic function for choosing object and EC
        #argmax samples from the [max (H(obj, ec)] list
        if h_process_type == "Deterministic":
            object_index,  ec_index = self.argmax_h(Q_matrix)
            # print(" ** h_mx[{}, {}]".format(object_index, ec_index))
            return objects[object_index], ecs[ec_index]
        # samples from [H(obj, ec)] list
        elif h_process_type == "Probabilistic":
            object_index, ec_index = self.sample_from_pdf(Q_matrix)
            return objects[object_index], ecs[ec_index]
        elif h_process_type == "Random":
            object_index, ec_index = self.random_from_Qmatrix(Q_matrix)
            return objects[object_index], ecs[ec_index]

        # worst case jsut return the first object and ec
        return (objects[0],ecs[0])

## --------------------------------------------------------- ##
## ---------------------- Main ----------------------------- ##
## --------------------------------------------------------- ##
if __name__ == "__main__":
    # testing the code inline
    obj_tf = tra.concatenate_matrices(
            tra.translation_matrix([-0.05, 0, 0.0]), tra.rotation_matrix(math.radians(170.0), [0, 0, 1]))
    ec = tra.concatenate_matrices(
            tra.translation_matrix([-0.05, 0, 0.0]), tra.rotation_matrix(math.radians(10.0), [0, 0, 1]))
    obj={}

    obj['frame'] = obj_tf
    obj['type'] = 'cucumber'
    strategy = 'WallGrasp'
    hand = 'RBOHandP24'

    # print obj_tf
    # print ec

    foo = multi_object_params()
    foo.load_object_params()
    h_val = h_val = foo.heuristic(obj,ec,strategy,hand)
    g_l = {"g1": h_val}
    print("h(g1)={}".format(h_val))

    #second EC
    ec = tra.concatenate_matrices(
        tra.translation_matrix([-0.05, 0, 0.0]),
        tra.rotation_matrix(math.radians(15.0), [0, 0, 1]),
        tra.rotation_matrix(math.radians(10.0), [0, 1, 0]))

    h_val = foo.heuristic(obj, ec, strategy, hand)
    print("h(g2)={}".format(h_val))
    g_l['g2'] = h_val

    print (g_l)
    argmax_h = foo.argmax_h(g_l)

    print("argmax(h) = {}".format(argmax_h))

    # h_val = foo.heuristic(obj, ec, strategy, hand)
    #
    #
    # print("H({}, {}, {}) = {}".format(obj["type"], strategy, hand, h_val))
