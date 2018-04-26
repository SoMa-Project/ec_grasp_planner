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
                print("data loaded")
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
    def pdf_object_ec(self, object, ec_frame):
        h = -1
        success = object['success']
        print object

        if object.get('angle',0):
            object_frame = object['frame']
            obj_x_axis = object_frame[0:3, 0]

            for idx, val in enumerate(object['angle']):
                ec_x_axis = ec_frame[0:3, 0]
                angle_epsilon = object['epsilon']

                if (math.fabs(angle_between(obj_x_axis, ec_x_axis) - math.radians(val))
                        <= math.radians(angle_epsilon)):
                    h = success[idx]
                    break
            # if the angle was not within the given bounded sets
            # take the last value from the list of success values
            if h == -1:
                h = success[-1]
            # if there are no other criteria for h
        else:
            h = success

        return h

## --------------------------------------------------------- ##
    #object-envirtionment-hand based heuristic valeu for grasping
    def heuristic(self, object, ec_frame, strategy, hand):

        object_params = self.data[object['type']][hand][strategy]
        object_params['frame'] = object['frame']
        h = 1

        h = h * self.pdf_object_strategy(object_params) * self.pdf_object_ec(object_params, ec_frame)

        return h

## --------------------------------------------------------- ##
    # find the max probability and if there are more than one return one randomly
    def argmax_h(self, h_matrix):
        # find max probablity in list



        # max_probability_indexes = np.argwhere(h_matrix == h_matrix.argmax())
        max_prob = h_matrix.max()
        h_matrix[h_matrix!=max_prob] = 0;

            #max(h_matrix.iteritems(), key=operator.itemgetter(1))[1]
        # print(max_probability)
        #
        # # select all items with max probability
        # max_probability_dict = {k:v for k,v in h_matrix.items() if float(v) == max_probability}
        # print (max_probability_dict)

        # select randomly one of the itmes
        ideces_sampled_max = self.sample_from_pdf(h_matrix)
        print (ideces_sampled_max)

        return ideces_sampled_max

## --------------------------------------------------------- ##
    # samples from a pdf dictionary where the values are normalized
    # returns the key of the sample
    def sample_from_pdf(self, pdf_matrix):

        pdf_array = np.reshape(pdf_matrix, pdf_matrix.shape[0]*pdf_matrix.shape[1] )

        # normalize pdf
        pdf_normalized = pdf_array/sum(pdf_array)

        # sample probabilistically
        # sampled_item = (np.random.choice(pdf_normalized.keys(), p=pdf_normalized.values()))

        sampled_item = (np.random.choice(len(pdf_normalized), p=pdf_normalized))

        return sampled_item // pdf_matrix.shape[1], sampled_item % pdf_matrix.shape[1]

## --------------------------------------------------------- ##
    # function called to process all objects and ECs
    # assumption1: all objects are the same type
    # objects is a dictionary with obilagorty keys: type, frame (in robot base frame)
    # ecs is a list of graph nodes (see geometry_graph)
    def process_objects_ecs(self, objects, ecs, graph_in_base, h_process_type="argmax", hand_type="RBOHandP24WAM"):

        self.load_object_params()
        h_matrix = np.zeros((len(objects),len(ecs)))

        # check if given hand exist
        for i,o in enumerate(objects):

            # check if the given hand type for this object is set in the yaml
            if not self.data[o["type"]].get(hand_type, 0):
                print("The givne hand {} and object {} has no parameters set in the yaml {}".format(hand_type, o["type"], self.file_name))
                return -1

            for j,ec in enumerate(ecs):
                # the ec frame must be in the same reference frame as the object
                ec_frame_in_base = graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform))
                h_matrix[i,j] = self.heuristic(o, ec_frame_in_base, ec.label, hand_type)

        if h_process_type == "argmax":
            object_index,  ec_index = self.argmax_h(h_matrix)
            return (objects[object_index], ecs[ec_index])

        elif h_process_type == "sample":
            object_index, ec_index = self.sample_from_pdf(h_matrix)
            return (objects[object_index], ecs[ec_index])


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
