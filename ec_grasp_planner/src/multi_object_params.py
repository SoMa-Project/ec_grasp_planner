#!/usr/bin/env python

import yaml
# import rospy
import math
import numpy as np
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


    #load parameters for hand-object-strategy
    def load_object_params(self):
        file = pkg_path + '/data/'+ self.file_name
        with open(file, 'r') as stream:
            try:
                self.data =yaml.load(stream)
                print("data loaded")
            except yaml.YAMLError as exc:
                print(exc)

    def pdf_object_strategy(self, object):
        # return 0 or 1 if strategy is applicable on the object
        # if there is a list of possible outcomes thant the strategy is applicable

        if isinstance(object['success'], list):
            return 1
        else:
            return object['success']

    def pdf_object_ec(self, object, ec):
        # return probability based on object and ec features
        h = -1
        success = object['success']
        print object

        if object.get('angle',0):
            object_frame = object['frame']
            obj_x_axis = object_frame[0:3, 0]

            for idx, val in enumerate(object['angle']):
                ec_x_axis = ec[0:3, 0]
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

    #object-envirtionment-hand based heuristic valeu for grasping
    def heuristic(self, object, ec, strategy, hand):

        object_params = self.data[object['type']][hand][strategy]
        object_params['frame'] = object['frame']
        h = 1

        h = h * self.pdf_object_strategy(object_params) * self.pdf_object_ec(object_params, ec)

        return h

    # find the max probability and if there are more than one return one randomly
    def argmax_h(self, grasp_dict):
        # find max probablity in list
        max_probability = max(grasp_dict.iteritems(), key=operator.itemgetter(1))[1]
        print(max_probability)

        # select all items with max probability
        max_probability_dict = {k:v for k,v in grasp_dict.items() if float(v) == max_probability}
        print (max_probability_dict)

        # select randomly one of the itmes
        max_random_item = self.sample_from_pdf(max_probability_dict)
        print (max_random_item)

        return max_random_item

    def sample_from_pdf(self, pdf_dict):

        # normalize pdf
        pdf_normalized = pdf_dict
        sum_ = sum(pdf_dict.values())
        pdf_normalized.update((x, y/sum_) for x, y in pdf_normalized.items())

        # sample probabilistically
        sampled_item = (np.random.choice(pdf_normalized.keys(), p=pdf_normalized.values()))
        return sampled_item

    def process_objects_ecs(self, objects, ecs, h_process_type = "argmax", hand_type = "RBOHandP24"):

        return (objects[0],ecs[0])

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
