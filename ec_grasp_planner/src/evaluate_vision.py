#!/usr/bin/env python

import sys
import rospy
from pregrasp_msgs import srv as vision_srv
import tf
from numpy.random.mtrand import choice
import numpy as np
import pandas as pd
import os
from tf import transformations as tra

def add_two_ints_client(x, y):
    rospy.wait_for_service('add_two_ints')
    try:
        add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)
        resp1 = add_two_ints(x, y)
        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [number_of_objects type_of_object number_of_experiments baseframe_name ifco_gt_frame_name vision_method]"%sys.argv[0]

def transform_msg_to_homogenous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]), tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))


if __name__ == "__main__":

    # arg 2 nr objects in ifco
    # arg 1 object typeifco_gt
    if len(sys.argv) == 7:
        nr_objects = int(sys.argv[1])
        object_type = str(sys.argv[2])
        nr_experiments = int(sys.argv[3])
        baseframe = str(sys.argv[4])
        ifco_gt_frame_name = sys.arvg[5]
        vision_method = sys.argv[6]
    else:
        print usage()
        sys.exit(1)
    print ("Experiment with object: {} nr of objects in IFCO: {}".format(object_type, nr_objects))



    rospy.init_node('vision_evaluator')
    rate = rospy.Rate(10.0)

    time_ = rospy.Time(0)

    # ifco_gt_frame_name = "camera_link"#""ifco_static"

    # get IFCO ground truth (TF listener)
    tf_listener = tf.TransformListener()
    tf_listener.waitForTransform(baseframe, ifco_gt_frame_name,
                                 time_,
                                 rospy.Duration(5.0))

    tmp = tf_listener.lookupTransform(baseframe, ifco_gt_frame_name, time_)
    ifco_in_base_grounde_truth = tmp

    file_name = "visionExperiment.csv"
    file_object = open(file_name, 'a')

    df_filename = 'vision_experiment_df.pickle'
    if not os.path.exists(df_filename):
        df = pd.DataFrame(columns=['object_type', 'ifco_ground_truth', 'ifco_observed', 'observed_object_count'])
    else
        df = pd.DataFrame.from_pickle(df_filename)

    ifco_gt_array = np.squeeze(np.asarray(ifco_in_base_grounde_truth))
    ifco_gt_str = np.array2string(ifco_gt_array, precision=2, separator=',', suppress_small=True)
    ifco_gt_str = ifco_gt_array.tolist()

    data = ""
    for i in range(0, nr_experiments):
        # call vision service
        call_vision = rospy.ServiceProxy('compute_ec_graph',
                                         vision_srv.ComputeECGraph)
        res = call_vision(object_type)
        nr_objects_observed = len(res.objects.objects)

        # get ifco frame
        graph = res.graph
        graph.header.stamp = time_


        slide_node =  [n for i, n in enumerate(graph.nodes) if n.label in ['Slide']]

        tf_listener.waitForTransform(baseframe, graph.header.frame_id,
                                    time_,
                                    rospy.Duration(5.0))

        graph_in_base_transform = tf_listener.asMatrix(baseframe, graph.header)
        ifco_observed = graph_in_base_transform .dot(transform_msg_to_homogenous_tf(slide_node[0].transform))

        # save data

        ifco_o_array =   np.squeeze(np.asarray(ifco_observed))
        # ifco_o_str = np.array2string(ifco_o_array, precision=2, separator=',', suppress_small=True, max_line_width=255)

        ifco_o_str = ifco_o_array.tolist()

        df.append({
            'object_type': object_type,
            'observed_object_count': nr_objects,
            'ifco_ground_truth': ifco_gt_array,
            'ifco_observed': ifco_observed
        })
        # data = object_type+ ", " + str(nr_objects) + ", " + str(ifco_gt_str) + ", " + str(nr_objects_observed) + ", " + str(ifco_o_str)
        # file_object.write(data)

        rate.sleep()

    df.to_pickle(df_filename)
    print ("result: {} ".format(data))
    file_object.close()