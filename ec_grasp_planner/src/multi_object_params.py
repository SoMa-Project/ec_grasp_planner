#!/usr/bin/env python

import yaml
import math
import numpy as np
from tf import transformations as tra
from geometry_graph_msgs.msg import Node, geometry_msgs
from tub_feasibility_check import srv as kin_check_srv
import rospy
from functools import partial

import rospkg

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


class PoseComponent(object):
    def __init__(self, x, y, z, w=None):
        # we have to explicitly convert to python floats since the later yaml conversion can't represent all
        # numpy.float64 values
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w) if w is not None else None

    def to_dict(self):
        d = {'x': self.x, 'y': self.y, 'z': self.z}
        if self.w is not None:
            d['w'] = self.w
        return d



class AlternativeBehavior:
    # TODO this class should be adapted if return value of the feasibility check changes (e.g. switch conditions)
    def __init__(self, feasibility_check_result):
        self.number_of_joints = len(feasibility_check_result.final_configuration)
        self.trajectory_steps = []
        for i in range(0, len(feasibility_check_result.trajectory), step=self.number_of_joints):
            self.trajectory_steps.append(feasibility_check_result.trajectory[i:i+self.number_of_joints])


class multi_object_params:
    def __init__(self, file_name="object_param.yaml"):
        self.file_name = file_name
        self.data = None
        self.stored_trajectories = {}

    def get_object_params(self):
        if self.data is None:
            self.load_object_params()
        return self.data

    # This function will return a dictionary, mapping every motion name (e.g. pre_grasp) to an alternative behavior
    # (e.g. a sequence of joint states) to the default hard-coded motion in the planner.py
    # If no such alternative behavior is defined the function returns None
    def get_alternative_behavior(self, object_idx, ec_index):
        if (object_idx, ec_index) not in self.stored_trajectories:
            return None
        return self.stored_trajectories[(object_idx, ec_index)]

    # ================================================================================================
    def transform_msg_to_homogenous_tf(self, msg):
        return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]),
                      tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))

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

    def check_kinematic_feasibility(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames,
                                    ifco_in_base_transform, handarm_params):

        object = objects[current_object_idx]
        ec_frame = all_ec_frames[current_ec_index]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        # TODO maybe move the kinematic stuff to separate file

        if strategy == 'SurfaceGrasp':
            # use kinematic checks
            # TODO create proxy; make it a persistent connection?

            # Code duplication from planner.py TODO put at a shared location

            if object['type'] in handarm_params['surface_grasp']:
                params = handarm_params['surface_grasp'][object['type']]
            else:
                params = handarm_params['surface_grasp']['object']
            # Set the initial pose above the object
            goal_ = np.copy(object_params['frame'])  # TODO: this should be support_surface_frame
            goal_[:3, 3] = tra.translation_from_matrix(object_params['frame'])
            goal_ = goal_.dot(params['hand_transform'])

            # the grasp frame is symmetrical - check which side is nicer to reach
            # this is a hacky first version for our WAM
            zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
            if goal_[0][0] < 0:
                goal_ = goal_.dot(zflip_transform)

            # hand pose above object
            pre_grasp_pose = goal_.dot(params['pregrasp_transform'])

            print(pre_grasp_pose)  # TODO bring that pose into a suitable format for the service call

            go_down_pose = pre_grasp_pose.dot(tra.translation_matrix([0, 0, -params['down_dist']])) # TODO multiplication order?

            # This list includes the checked motions in order (They have to be sequential!)
            checked_motions = ["pre_grasp", "go_down"]
            # The goal poses of the respective motions in op-space (index has to match index of checked_motions)
            goals = [pre_grasp_pose, go_down_pose]
            # gotoview joint config (copied from gui.py)
            # TODO read this from a central point (to ensure it is always the same parameter in gui and here)
            curr_start_config = [0.457929, 0.295013, -0.232804, 2.59226, 1.25715, 1.50907, -0.616263]
            # This variable is used to determine if all returned status flags are 1 (original trajectory feasible)
            status_sum = 0
            # initialize stored trajectories for the given object
            self.stored_trajectories[(current_object_idx, current_ec_index)] = {}

            for motion, curr_goal in zip(checked_motions, goals):

                manifold_name = motion +'_manifold'

                bounding_boxes = []
                for obj in objects:

                    obj_trans, obj_rot = multi_object_params.transform_to_python_pose(obj['frame'])
                    bounding_boxes.append({
                        'box': {
                            'type': 0,
                            'dimensions': [obj['bounding_box'].x, obj['bounding_box'].y, obj['bounding_box'].z]
                        },
                        'pose': {
                            'position': obj_trans.to_dict(),
                            'orientation': obj_rot.to_dict(),
                        }
                    })

                ifco_trans, ifco_rot = multi_object_params.transform_to_python_pose(ifco_in_base_transform)
                goal_trans, goal_rot = multi_object_params.transform_to_python_pose(curr_goal)

                geometry_msgs.msg.Pose() # TODO USE or REMOVE

                args = {
                    'initial_configuration': curr_start_config,
                    'goal_pose': {
                        'position': goal_trans.to_dict(),
                        'orientation': goal_rot.to_dict(),
                    },
                    'ifco_pose': {
                        'position': ifco_trans.to_dict(),
                        'orientation': ifco_rot.to_dict(),
                    },
                    'bounding_boxes_with_poses': bounding_boxes,
                    'min_position_deltas': params[manifold_name]['min_position_deltas'],
                    'max_position_deltas': params[manifold_name]['max_position_deltas'],
                    'min_orientation_deltas': params[manifold_name]['min_orientation_deltas'],
                    'max_orientation_deltas': params[manifold_name]['max_orientation_deltas'],
                    # TODO currently we only allow to touch the object to be grasped during a surface grasp, is that really desired? (what about a really crowded ifco)
                    'allowed_collisions': [{'type': 1, 'box_id': current_object_idx, 'terminate_on_collision': True},
                                           {'type': 2, 'constraint_name': 'bottom', 'terminate_on_collision': False}]
                }

                check_kinematics = rospy.ServiceProxy('/check_kinematics', kin_check_srv.CheckKinematics)
                print(args)
                print("Call check kinematics. Arguments: \n" + yaml.safe_dump(args))

                res = check_kinematics(initial_configuration=curr_start_config,
                                       goal_pose= geometry_msgs.msg.Pose())# TODO call actual method

                if res.status == 0:
                    # trajectory is not feasible and no alternative was found, directly return 0
                    return 0

                elif res.status == 2:
                    # original trajectory is not feasible, but alternative was found => save it
                    self.stored_trajectories[(current_object_idx, current_ec_index)][motion] = AlternativeBehavior(res)
                    curr_start_config = res.final_configuration

                elif res.status == 1:
                    # original trajectory is feasible. If all checked motions remain feasible, status_sum will be used
                    # to signal that the original HA should not be touched.
                    status_sum += 1

                else:
                    raise ValueError(
                        "check_kinematics: No handler for result status of {} implemented".format(res.status))

            if status_sum == len(checked_motions):
                # all results had status 1, this means we can generate a HA without adding any special joint controllers
                # In order to signal that we throw away all generated alternative trajectories.
                self.stored_trajectories[(current_object_idx, current_ec_index)] = None

        else:
            # TODO implement other strategies
            raise ValueError("Kinematics checks are currently only supported for surface grasps")

        return self.pdf_object_strategy(object_params) * self.pdf_object_ec(object_params, ec_frame,
                                                                            strategy)



    def black_list_risk_regions(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames,
                                ifco_in_base_transform):

        object = objects[current_object_idx]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        zone_fac = self.black_list_unreachable_zones(object, object_params, ifco_in_base_transform, strategy)
        wall_fac = self.black_list_walls(current_ec_index, all_ec_frames, strategy)

        return zone_fac * wall_fac

    @staticmethod
    def transform_to_pose(in_transform):
        # convert 4x4 matrix to trans + rot
        scale, shear, angles, translation, persp = tra.decompose_matrix(in_transform)
        orientation_quat = tra.quaternion_from_euler(angles[0], angles[1], angles[2])
        return translation, orientation_quat

    @staticmethod
    def transform_to_python_pose(in_transform):
        trans, rot = multi_object_params.transform_to_pose(in_transform)
        return PoseComponent(trans[0], trans[1], trans[2]), PoseComponent(rot[0], rot[1], rot[2], rot[3])

    def reset_kinematic_checks_information(self):
        self.stored_trajectories = {}

## --------------------------------------------------------- ##
    # object-environment-hand based heuristic, q_value for grasping
    def heuristic(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames, ifco_in_base_transform, handarm_params):

        object = objects[current_object_idx]

        ec_frame = all_ec_frames[current_ec_index]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        use_kinematic_checks = rospy.get_param("feasibility_check/active", default=True)
        if use_kinematic_checks:
            feasibility_fun = partial(self.check_kinematic_feasibility, current_object_idx, objects, current_ec_index,
                                      strategy, all_ec_frames, ifco_in_base_transform, handarm_params)
        else:
            feasibility_fun = partial(self.black_list_risk_regions, current_object_idx, objects, current_ec_index,
                                      strategy, all_ec_frames, ifco_in_base_transform)

        q_val = 1
        q_val = q_val * \
            self.pdf_object_strategy(object_params) * \
            self.pdf_object_ec(object_params, ec_frame, strategy) * \
            feasibility_fun()

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
    def process_objects_ecs(self, objects, ecs, graph_in_base, ifco_in_base_transform, h_process_type="Deterministic",
                            handarm_parameters=None): # TODO replace default

        # print("object: {}, \n ecs: {} \n graphTF: {}, h_process: {}".format(objects, ecs, graph_in_base, h_process_type))
        # print("ec type: {}".format(type(ecs[0])))
        # load parameter file
        self.load_object_params()
        self.reset_kinematic_checks_information()

        Q_matrix = np.zeros((len(objects), len(ecs)))

        # iterate through all objects
        for i, o in enumerate(objects):

            # check if the given hand type for this object is set in the yaml
            # print ("object type: {}".format(o["type"]))

            if not self.data[o["type"]]:
                print("The given object {} has no parameters set in the yaml {}".format(o["type"], self.file_name))
                return -1

            all_ec_frames = []
            for j, ec in enumerate(ecs):
                all_ec_frames.append(graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform)))
                print("ecs:{}".format(graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform))))

            for j, ec in enumerate(ecs):
                # the ec frame must be in the same reference frame as the object
                ec_frame_in_base = graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform))
                Q_matrix[i,j] = self.heuristic(i, objects, j, ec.label, all_ec_frames, ifco_in_base_transform,
                                               handarm_parameters)

        # print (" ** h_mx = {}".format(Q_matrix))

        # select heuristic function for choosing object and EC
        #argmax samples from the [max (H(obj, ec)] list
        if h_process_type == "Deterministic":
            object_index,  ec_index = self.argmax_h(Q_matrix)
            print(" ** h_mx[{}, {}]".format(object_index, ec_index))
            print(" ** h_mx[{}, {}]".format(object_index, ecs[ec_index]))
            return object_index, ec_index
        # samples from [H(obj, ec)] list
        elif h_process_type == "Probabilistic":
            object_index, ec_index = self.sample_from_pdf(Q_matrix)
            return object_index, ec_index
        elif h_process_type == "Random":
            object_index, ec_index = self.random_from_Qmatrix(Q_matrix)
            return object_index, ec_index

        # worst case just return the first object and ec
        return 0, 0


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
