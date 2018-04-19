#!/usr/bin/env python

import yaml
# import rospy
import math
import numpy as np
from tf import transformations as tra

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

    def heuristic(self, object, ec, strategy, hand):

        object_params = self.data[object['type']][hand][strategy]
        object_params['frame'] = object['frame']
        h = 1

        h = h * self.pdf_object_strategy(object_params) * self.pdf_object_ec(object_params, ec)

        return h


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

    print obj_tf
    print ec

    foo = multi_object_params()
    foo.load_object_params()
    h_val = foo.heuristic(obj,ec,strategy,hand)

    print("H({}, {}, {}) = {}".format(obj["type"], strategy, hand, h_val))
