#!/usr/bin/env python

import yaml
import math
import numpy as np
from tf import transformations as tra
from geometry_graph_msgs.msg import Node, geometry_msgs
from tub_feasibility_check import srv as kin_check_srv
from tub_feasibility_check.msg import BoundingBoxWithPose, AllowedCollision
from tub_feasibility_check.srv import CheckKinematicsTabletopResponse
from shape_msgs.msg import SolidPrimitive
import rospy
from functools import partial
import planner_utils as pu
import tf2_ros


import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')

tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

class EnvironmentalConstraint:
    def __init__(self, transform, label):
        self.transform = transform
        self.label = label


class AlternativeBehavior:
    # TODO this class should be adapted if return value of the feasibility check changes (e.g. switch conditions)
    def __init__(self, feasibility_check_result, init_conf):
        self.number_of_joints = len(feasibility_check_result.final_configuration)
        self.trajectory_steps = []
        for i in range(0, len(feasibility_check_result.trajectory), self.number_of_joints):
            self.trajectory_steps.append(feasibility_check_result.trajectory[i:i+self.number_of_joints])

        if np.allclose(init_conf, self.trajectory_steps[0]):
            rospy.logwarn("Initial configuration {0} is first point in trajectory".format(init_conf))
            # :1 = skip the initial position TODO remove if relative is used!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.trajectory_steps = self.trajectory_steps[1:]

    def assert_that_initial_config_not_included(self, init_conf):
        if np.allclose(init_conf, self.trajectory_steps[0]):
            raise ValueError("Initial configuration {0} is first point in trajectory".format(init_conf))

    def get_trajectory(self):
        print("get_trajectory LEN:", len(self.trajectory_steps))
        return np.transpose(np.array(self.trajectory_steps))


class multi_object_params:
    def __init__(self, file_name="object_param.yaml"):
        self.file_name = file_name
        self.data = None
        self.stored_trajectories = {}

    def get_object_params(self):
        if self.data is None:
            self.load_object_params()
        return self.data

    # This function will return a dictionary, mapping a motion name (e.g. pre_grasp) to an alternative behavior
    # (e.g. a sequence of joint states) to the default hard-coded motion in the planner.py (in case it was necessary to
    # generate). If for the given object-ec-pair no such alternative behavior was created, this function returns None.
    def get_alternative_behavior(self, object_idx, ec_index):
        print(self.stored_trajectories)
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
                diff_angle = math.fabs(pu.angle_between(obj_x_axis, ec_x_axis) - math.radians(val))
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

        # favor edges that are perpendicular to long object axis (x-axis)
        if strategy in ["EdgeGrasp"]:
            e_x_object = object_frame[:3, 0]
            # e_y_object = object_frame[:3, 1]
            e_y_ec = ec_frame[:3, 1]

            q_val *= abs(np.dot(e_y_ec, e_x_object))

        # distance form EC (wall end edge)
        # this is the tr from object_frame to ec_frame in object frame
        if strategy in ["WallGrasp", "EdgeGrasp"]:
            delta = np.linalg.inv(ec_frame).dot(object_frame)
            # this is the distance between object and EC
            dist = abs(delta[2, 3]) # TODO why was this never a problem before?
            # include distance to q_val, longer distance decreases q_val
            q_val = q_val * (1/abs(dist))

        return q_val

    def black_list_walls(self, current_ec_index, all_ec_frames, strategy):

        print(":::A1")
        if strategy not in ["WallGrasp", "EdgeGrasp"]:
            return 1
        # this function will blacklist all walls except
        # the one on th right side of the robot
        # y coord is the smallest

        print(":::A2")
        if all_ec_frames[current_ec_index][1,3] > 0:
            print(":::A3")
            return 0

        min_y = 10000
        min_y_index = 0

        for i, ec in enumerate(all_ec_frames):
            if min_y > ec[1,3]:
                min_y = ec[1,3]
                min_y_index = i

        if min_y_index == current_ec_index:
            print(":::A4")
            return 1
        else:
            print(":::A5")
            return 0

    def black_list_edges(self, current_ec_index, all_ec_frames, strategy, table_frame, objects, current_object_idx, handarm_params):

        if strategy not in ["EdgeGrasp"]:
            return 1

        obj = objects[current_object_idx]

        if obj['type'] in handarm_params['edge_grasp']:
            params = handarm_params['edge_grasp'][obj['type']]
        else:
            params = handarm_params['edge_grasp']['object']

        e_x_table = table_frame[:3, 0]
        e_y_table = table_frame[:3, 1]

        ec = all_ec_frames[current_ec_index]

        # e_x_ec = ec[:3, 0]
        e_y_ec = ec.transform[:3, 1]

        # we only allow the edge below the table x-axis and the edge left of the table y-axis to be considered
        if np.dot(e_y_ec, e_x_table) < -0.8 or np.dot(e_y_ec, e_y_table) < -0.8:
            value = 0

        else:
            value = 1



        # invert value if pushing
        if params['sliding_direction'] == 1:
            value = 1 - value

        return value

    def black_list_unreachable_zones(self, object, object_params, ifco_in_base_transform, strategy):

        # this function will blacklist out of reach zones for wall and surface grasp
        if strategy not in ["WallGrasp", "SurfaceGrasp"]:
            return 1

        object_min = object_params['min']
        object_max = object_params['max']
        object_frame = object['frame']

        object_in_ifco_frame = ifco_in_base_transform.dot(object_frame)

        if object_in_ifco_frame[0,3] > object_min[0]  \
            and object_in_ifco_frame[0,3] < object_max[0] \
            and object_in_ifco_frame[1,3] > object_min[1] \
            and object_in_ifco_frame[1,3] < object_max[1]:
            return 1
        else:
            return 0

    @staticmethod
    def get_matching_ifco_wall(ifco_in_base_transform, ec_frame):

        ec_to_world = ifco_in_base_transform.dot(ec_frame)
        ec_z_axis_in_world = ec_to_world.dot(np.array([0, 0, 1, 1]))[:3]
        ec_x_axis_in_world = ec_to_world.dot(np.array([1, 0, 0, 1]))[:3]

        # one could also check for dot-product = 0 instead of using the x-axis but this is prone to numeric issues.
        if ec_z_axis_in_world.dot(np.array([1, 0, 0])) > 0 and ec_x_axis_in_world.dot(np.array([0, 1, 0])) > 0:
            print("FOUND_EC south")
            return 'south'
        elif ec_z_axis_in_world.dot(np.array([1, 0, 0])) < 0 and ec_x_axis_in_world.dot(np.array([0, 1, 0])) < 0:
            print("FOUND_EC north")
            return 'north'
        elif ec_z_axis_in_world.dot(np.array([0, 1, 0])) < 0:
            print("FOUND_EC west")
            return 'west'
        else:
            print("FOUND_EC east")
            return 'east'

    ## have the possibility to only check feasibility for ecs that pass the heuristics
    def heuristics_and_check_kinematic_feasibility(self, current_ec_index, all_ecs, strategy, ifco_in_base_transform,
                                                   current_object_idx, objects, handarm_params):

        if strategy == 'EdgeGrasp':

            value = self.black_list_edges(current_ec_index, all_ecs, strategy, ifco_in_base_transform, objects, current_object_idx, handarm_params)

            if value == 0:
                return value

        return self.check_kinematic_feasibility(current_object_idx, objects, current_ec_index, all_ecs,
                                                ifco_in_base_transform, handarm_params)



    # TODO move that to a separate file?
    def check_kinematic_feasibility(self, current_object_idx, objects, current_ec_index, all_ecs,
                                    ifco_in_base_transform, handarm_params):

        strategy = all_ecs[current_ec_index].label
        ec_frame = all_ecs[current_ec_index].transform

        object = objects[current_object_idx]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        # This list includes the checked motions in order (They have to be sequential!)
        checked_motions = []
        # The goal poses of the respective motions in op-space (index has to match index of checked_motions)
        goals = []
        # The collisions that are allowed in message format per motion
        allowed_collisions = {}

        # The initial joint configuration (goToView config) This is not used right know, since view->init is not checked
        # curr_start_config = rospy.get_param('planner_gui/robot_view_position') # TODO check view->init config

        # The edges in the scene
        edges = [multi_object_params.transform_to_pose_msg(e.transform) for e in all_ecs if e.label == "EdgeGrasp"]

        # The backup table frame that is used in case we don't create the table from edges
        #table_frame = None
        #table_pose = geometry_msgs.msg.Pose()
        table_frame = ifco_in_base_transform # no need to invert
        # flip z-axis
        table_frame = table_frame.dot(tra.rotation_matrix(math.radians(180), [1, 0, 0]))

        table_pose = multi_object_params.transform_to_pose_msg(table_frame)

        # TODO maybe move the kinematic stuff to separate file

        if strategy == 'SurfaceGrasp':
            # use kinematic checks
            # TODO create proxy; make it a persistent connection?

            # Code duplication from planner.py TODO put at a shared location

            if object['type'] in handarm_params['surface_grasp']:
                params = handarm_params['surface_grasp'][object['type']]
            else:
                params = handarm_params['surface_grasp']['object']

            # table_frame = tra.inverse_matrix(ifco_in_base_transform)  # ec_frame (does not have the correct orientation)

            # Since the surface grasp frame is at the object center we have to translate it in z direction
            #table_frame[2, 3] = table_frame[2, 3] - object['bounding_box'].z / 2.0

            # Convert table frame to pose message
            # table_pose = multi_object_params.transform_to_pose_msg(table_frame)

            object_frame = object_params['frame']

            #print("DBG_FRAME 1", multi_object_params.transform_to_pose_msg(object_frame))

            goal_ = np.copy(object_frame)
            # we rotate the frame to align the hand with the long object axis and the table surface
            x_axis = object_frame[:3, 0]
            z_axis = table_frame[:3, 2]
            y_axis = np.cross(z_axis, x_axis)

            goal_[:3, 0] = x_axis
            goal_[:3, 1] = y_axis / np.linalg.norm(y_axis)
            goal_[:3, 2] = z_axis

            print("DBG_FRAME 1", pu.tf_dbg_call_to_string(goal_, "dbg1"))

            # In the disney use case the z-axis points downwards, but the planner compensates for that already by
            # flipping the support_surface_frame. Since the feasibilty_checker assumes the z-axis for the disney use
            # case flipped as well, we need to apply the flip afterwards, when creating the pre-grasp pose
            # TODO get rid of this mess (change convention in feasibility checker)
            x_flip_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(180.0), [1, 0, 0]))
            goal_ = goal_.dot(x_flip_transform)

            goal_ = goal_.dot(params['hand_transform'])

            print("DBG_FRAME 2", pu.tf_dbg_call_to_string(goal_, "dbg2"))

            # Set the initial pose above the object
            #goal_ = np.copy(object_params['frame'])  # TODO: this should be support_surface_frame
            #goal_[:3, 3] = tra.translation_from_matrix(object_params['frame'])
            #goal_ = goal_.dot(params['hand_transform'])

            # the grasp frame is symmetrical - check which side is nicer to reach
            # this is a hacky first version for our WAM
            zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
            if goal_[0][0] < 0:
                goal_ = goal_.dot(zflip_transform)

            print("DBG_FRAME 3",  multi_object_params.transform_to_pose_msg(goal_))

            # hand pose above object
            pre_grasp_pose = goal_.dot(params['pregrasp_transform_alt'])

            # down_dist = params['down_dist']  #  dist lower than ifco bottom: behavior of the high level planner
            # dist = z difference to object centroid (both transformations are w.r.t. to world frame
            # (more realistic behavior since we have to touch the object for a successful grasp)
            down_dist = pre_grasp_pose[2, 3] - object_params['frame'][2, 3]  # get z-translation difference

            # add safety distance above object
            down_dist -= params['safety_distance_above_object']

            # goal pose for go down movement
            go_down_pose = tra.translation_matrix([0, 0, -down_dist]).dot(pre_grasp_pose)

            post_grasp_pose = params['post_grasp_transform'].dot(go_down_pose)  # TODO it would be better to allow relative motion as goal frames

            checked_motions = ["pre_grasp", "go_down"]#, "post_grasp_rot"] ,go_up, go_drop_off  # TODO what about remaining motions? (see wallgrasp)

            goals = [pre_grasp_pose, go_down_pose]#, post_grasp_pose]

            # TODO what about using the bounding boxes as for automatic goal manifold calculation?

            # Take orientation of object but translation of pre grasp pose
            pre_grasp_pos_manifold = np.copy(object_params['frame'])
            pre_grasp_pos_manifold[:3, 3] = tra.translation_from_matrix(pre_grasp_pose)

            go_down_pos_manifold = np.copy(object_params['frame'])
            go_down_pos_manifold[:3, 3] = tra.translation_from_matrix(go_down_pose)

            goal_manifold_frames = {
                'pre_grasp': pre_grasp_pos_manifold,

                # Use object frame for resampling
                'go_down': go_down_pos_manifold  # TODO change that again to go_down_pose!? <-- YES?!?!
            }

            goal_manifold_orientations = {
                # use hand orientation
                'pre_grasp': tra.quaternion_from_matrix(pre_grasp_pose),

                # Use object orientation
                'go_down': tra.quaternion_from_matrix(go_down_pose), #tra.quaternion_from_matrix(object_params['frame'])  # TODO use hand orietation instead?
            }

            # override initial robot configuration
            # TODO also check gotToView -> params['initial_goal'] (requires forward kinematics, or change to op-space)
            curr_start_config = params['initial_goal']

            print("ALLOWED COLLISIONS:", "box_" + str(current_object_idx), 'bottom')

            # The collisions that are allowed per motion in message format
            # TODO currently we only allow to touch the object to be grasped and the ifco bottom during a surface grasp, is that really desired? (what about a really crowded ifco)
            allowed_collisions = {
                # no collisions are allowed during going to pre_grasp pose
                'pre_grasp': [],

                'go_down': [AllowedCollision(type=AllowedCollision.BOUNDING_BOX,
                                             # changed to not-terminating and not-required since collision model is not
                                             # accurate enough at the moment
                                             box_id=current_object_idx, terminating=False, required=False),
                            AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                             constraint_name='tabletop', terminating=False)],

                # TODO also account for the additional object in a way?
                'post_grasp_rot': [AllowedCollision(type=AllowedCollision.BOUNDING_BOX,
                                                    box_id=current_object_idx, terminating=True),
                                   AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                                    constraint_name='tabletop', terminating=False)]
            }

        elif strategy == "WallGrasp":

            blocked_ecs = [0, 1, 2,  4] # TODO remove
            if current_ec_index in blocked_ecs:
                return 0

            if object['type'] in handarm_params['wall_grasp']:
                params = handarm_params['wall_grasp'][object['type']]
            else:
                params = handarm_params['wall_grasp']['object']

            # hand pose above and behind the object
            pre_approach_transform = params['pre_approach_transform']

            wall_frame = np.copy(ec_frame)
            wall_frame[:3, 3] = tra.translation_from_matrix(object_params['frame'])
            # apply hand transformation
            ec_hand_frame = wall_frame.dot(params['hand_transform'])

            #ec_hand_frame = (ec_frame.dot(params['hand_transform']))
            pre_approach_pose = ec_hand_frame.dot(pre_approach_transform)

            # goal pose for go down movement

            # down_dist = params['down_dist']  #  dist lower than ifco bottom: behavior of the high level planner
            # dist = z difference to ifco bottom minus hand frame offset (dist from hand frame to collision point)
            # (more realistic behavior since we have a force threshold when going down to the bottom)
            bounded_lift_down_dist = pre_approach_pose[2, 3] - tra.inverse_matrix(ifco_in_base_transform)[2, 3]
            hand_frame_to_bottom_offset = 0.07  # 7cm TODO maybe move to handarm_parameters.py
            bounded_lift_down_dist = min(params['down_dist'], bounded_lift_down_dist-hand_frame_to_bottom_offset)
            ## TODO add back bounded_lift_down_dist when HA_mnager can stop jont trajectory with FT
            go_down_pose = tra.translation_matrix([0, 0, -0.1]).dot(pre_approach_pose)

            # pose after lifting. This is somewhat fake, since the real go_down_pose will be determined by
            # the FT-Switch during go_down and the actual lifted distance by the TimeSwitch (or a pose switch in case
            # the robot allows precise small movements) TODO better solution?
            fake_lift_up_dist = np.min([params['lift_dist'], 0.01])  # 1cm
            lift_hand_pose = tra.translation_matrix([0, 0, fake_lift_up_dist]).dot(go_down_pose)

            dir_wall = tra.translation_matrix([0, 0, -params['sliding_dist']])
            # TODO sliding_distance should be computed from wall and hand frame.
            # slide direction is given by the normal of the wall
            wall_frame = np.copy(ec_frame)
            dir_wall[:3, 3] = wall_frame[:3, :3].dot(dir_wall[:3, 3])

            # normal goal pose behind the wall
            slide_to_wall_pose = dir_wall.dot(lift_hand_pose)

            # now project it into the wall plane!
            z_projection = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 1]])

            to_wall_plane_transform = wall_frame.dot(z_projection.dot(tra.inverse_matrix(wall_frame).dot(slide_to_wall_pose)))
            slide_to_wall_pose[:3, 3] = tra.translation_from_matrix(to_wall_plane_transform)

            # TODO remove code duplication with planner.py (refactor code snippets to function calls) !!!!!!!

            checked_motions = ['pre_grasp', 'go_down', 'lift_hand', 'slide_to_wall'] # TODO overcome problem of FT-Switch after go_down

            goals = [pre_approach_pose, go_down_pose, lift_hand_pose, slide_to_wall_pose] # TODO see checked_motions

            # Take orientation of object but translation of pre grasp pose
            pre_grasp_pos_manifold = np.copy(object_params['frame'])
            pre_grasp_pos_manifold[:3, 3] = tra.translation_from_matrix(pre_approach_pose)

            slide_pos_manifold = np.copy(slide_to_wall_pose)

            goal_manifold_frames = {
                'pre_grasp': pre_grasp_pos_manifold,

                # Use object frame for sampling
                'go_down': np.copy(go_down_pose),

                'lift_hand': np.copy(lift_hand_pose),  # should always be the same frame as go_down # TODO use world orientation?

                # Use wall frame for sampling. Keep in mind that the wall frame has different orientation, than world.
                'slide_to_wall': slide_pos_manifold,
            }

            goal_manifold_orientations = {
                # use hand orientation
                'pre_grasp': tra.quaternion_from_matrix(pre_approach_pose),

                # Use object orientation
                'go_down': tra.quaternion_from_matrix(go_down_pose),  # TODO use hand orietation instead?

                # should always be the same orientation as go_down
                'lift_hand': tra.quaternion_from_matrix(lift_hand_pose),

                # use wall orientation
                'slide_to_wall': tra.quaternion_from_matrix(wall_frame),
            }

            # override initial robot configuration
            # TODO also check gotToView -> params['initial_goal'] (requires forward kinematics, or change to op-space)
            curr_start_config = params['initial_goal']

            allowed_collisions = {

                # 'init_joint': [],

                # no collisions are allowed during going to pre_grasp pose
                'pre_grasp': [],

                # Only allow touching the bottom of the ifco
                'go_down': [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                             terminating=False),
                            ],

                'lift_hand': [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                               terminating=False),
                              ],

                # TODO also allow all other obejcts to be touched during sliding motion
                'slide_to_wall': [
                                            # Allow all other objects to be touched as well
                                            # (since hand will go through them in simulation) TODO desired behavior?
                                            AllowedCollision(type=AllowedCollision.BOUNDING_BOX, box_id=obj_idx,
                                                             terminating=False, required=obj_idx == current_object_idx)
                                            for obj_idx in range(0, len(objects))
                                 ]
                + [
                                     AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                                      constraint_name=multi_object_params.get_matching_ifco_wall(
                                                          ifco_in_base_transform, ec_frame),
                                                      terminating=False),

                                     AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                                      terminating=False),
                  ],

            }

        elif strategy == "EdgeGrasp":  # TODO get rid of code redundancy

            if object['type'] in handarm_params['edge_grasp']:
                params = handarm_params['edge_grasp'][object['type']]
            else:
                params = handarm_params['edge_grasp']['object']

            hand_transform = params['hand_transform']
            pre_approach_transform = params['pre_approach_transform_alt']
            slide_transform = params['slide_transform_alt']
            palm_edge_offset = params['palm_edge_offset_alt']

            # This is the frame of the edge we are going for
            edge_frame = np.copy(ec_frame)

            # In the disney use case the z-axis of the table points downwards and the z-axis of the edge lays inside
            # the plane, but the planner compensates for that already by rotating the edge_frame.
            # Since the feasibilty_checker assumes the z-axis for the disney use
            # case flipped as well, we need to apply the flip afterwards, when creating the pre-grasp pose
            # TODO get rid of this mess (change convention in feasibility checker)
            x_rot_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(-90.0), [1, 0, 0]))
            edge_frame = edge_frame.dot(x_rot_transform)

            print("EDGE_MULTI", pu.tf_dbg_call_to_string(edge_frame, "EDGE_MULTI"))
            print("OBJ_MULTI", pu.tf_dbg_call_to_string(object_params['frame'], "OBJ_MULTI"))


            # lets force the object's x-axis to point away from the robot
            object_pose = np.copy(object_params['frame'])

            object_x_axis = object_pose[:2, 0]
            object_x_axis /= np.linalg.norm(object_x_axis)

            table_x_axis = ifco_in_base_transform[:2, 0]
            table_x_axis /= np.linalg.norm(table_x_axis)

            table_y_axis = ifco_in_base_transform[:2, 1]
            table_y_axis /= np.linalg.norm(table_y_axis)

            c = np.dot(table_x_axis, object_x_axis.T)
            s = np.dot(table_y_axis, object_x_axis.T)

            angle = math.degrees(math.atan2(s, c)) % 360

            if angle > 135 and angle < 315:
                object_pose = object_pose.dot(tra.rotation_matrix(math.radians(180), [0, 0, 1]))


            # this is the EC frame. It is positioned like object and oriented to the edge?
            initial_slide_frame = np.copy(edge_frame)

            # flip orientation if pushing instead of pulling
            if params['sliding_direction'] == 1:
                initial_slide_frame = np.dot(initial_slide_frame, tra.rotation_matrix(math.radians(180), [0, 1, 0]))

            initial_slide_frame[:3, 3] = tra.translation_from_matrix(object_pose)
            # apply hand transformation
            # hand on object fingers pointing toward the edge
            initial_slide_frame = (initial_slide_frame.dot(hand_transform))

            # the pre-approach pose should be:
            # - floating above the object
            # -- position above the object depend the relative location of the edge to the object and the object bounding box
            #   TODO define pre_approach_transform(egdeTF, objectTF, objectBB)
            # - fingers pointing toward the edge (edge x-axis orthogonal to palm frame x-axis)
            # - palm normal perpendicualr to surface normal
            # pre_approach_pose = initial_slide_frame.dot(pre_approach_transform)
            pre_approach_pose = object_pose.dot(pre_approach_transform)

            print("PREEE", pu.tf_dbg_call_to_string(pre_approach_pose, "PRE1"))

            # down_dist = params['down_dist']  #  dist lower than ifco bottom: behavior of the high level planner
            # dist = z difference to object centroid (both transformations are w.r.t. to world frame
            # (more realistic behavior since we have to touch the object for a successful grasp)
            down_dist = pre_approach_pose[2, 3] - object_pose[2, 3]  # get z-translation difference

            # add safety distance above object
            down_dist -= params['safety_distance_above_object']

            # goal pose for go down movement
            go_down_pose = tra.translation_matrix([0, 0, -down_dist]).dot(pre_approach_pose)

            # goal pose for sliding
            # 4. Go towards the edge to slide object to edge
            # this is the tr from initial_slide_frame to edge frame in initial_slide_frame
            delta = np.linalg.inv(edge_frame).dot(initial_slide_frame)

            # this is the distance to the edge, given by the z axis of the edge frame
            # add some extra forward distance to avoid grasping the edge of the table palm_edge_offset
            dist = delta[2, 3] + np.sign(delta[2, 3]) * palm_edge_offset

            # handPalm pose on the edge right before grasping
            hand_on_edge_pose = initial_slide_frame.dot(tra.translation_matrix([dist, 0, 0]))

            # direction toward the edge and distance without any rotation in worldFrame
            dir_edge = np.identity(4)
            # relative distance from initial slide position to final hand on the edge position

            distance_to_edge = (hand_on_edge_pose - initial_slide_frame) * params['sliding_direction']
            # TODO might need tuning based on object and palm frame
            dir_edge[:3, 3] = tra.translation_from_matrix(distance_to_edge)
            # no lifting motion applied while sliding
            dir_edge[2, 3] = 0

            # slide_to_edge_pose = dir_edge.dot(go_down_pose)
            slide_to_edge_pose = dir_edge.dot(initial_slide_frame)
            ## we want the same roll, pitch orientation as as pre_approach_pose and go_down_pose but be oriented as edge
            ## frame in terms of yaw
            R_pre_approach = np.identity(4)
            R_pre_approach[:3, :3] = pre_approach_transform[:3, :3]
            slide_to_edge_pose = slide_to_edge_pose.dot(R_pre_approach)

            slide_to_edge_pose = slide_to_edge_pose.dot(slide_transform)

            # slide_to_edge_pose = slide_to_edge_pose.dot(tra.rotation_matrix(0.82, [0, 1, 0]))

            checked_motions = ['pre_grasp', 'go_down',
                               'slide_to_edge']  # TODO overcome problem of FT-Switch after go_down

            goals = [pre_approach_pose, go_down_pose, slide_to_edge_pose]  # TODO see checked_motions

            # Take orientation of object but translation of pre grasp pose
            pre_grasp_pos_manifold = np.copy(object_pose)
            pre_grasp_pos_manifold[:3, 3] = tra.translation_from_matrix(pre_approach_pose)

            go_down_manifold = np.copy(object_pose)
            go_down_manifold[:3, 3] = tra.translation_from_matrix(go_down_pose)

            slide_to_edge_manifold = np.copy(object_pose)
            slide_to_edge_manifold[:3, 3] = tra.translation_from_matrix(slide_to_edge_pose)


            goal_manifold_frames = {
                'pre_grasp': pre_grasp_pos_manifold,

                # Use object frame for sampling
                'go_down': go_down_manifold,

                # Use wall frame for sampling. Keep in mind that the wall frame has different orientation, than world.
                'slide_to_edge': slide_to_edge_manifold,
            }

            goal_manifold_orientations = {
                # use hand orientation
                'pre_grasp': tra.quaternion_from_matrix(pre_approach_pose),

                # Use object orientation
                'go_down': tra.quaternion_from_matrix(go_down_pose), # TODO use hand orientation instead?

                # use wall orientation
                'slide_to_edge': tra.quaternion_from_matrix(slide_to_edge_pose),
            }

            # override initial robot configuration
            # TODO also check gotToView -> params['initial_goal'] (requires forward kinematics, or change to op-space)
            curr_start_config = params['initial_goal']

            allowed_collisions = {

                # 'init_joint': [],

                # no collisions are allowed during going to pre_grasp pose
                'pre_grasp': [],

                # Only allow touching the table
                'go_down': [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='tabletop',
                                             terminating=False),
                            ] + [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='box_0',
                                             terminating=False),
                            ],

                'slide_to_edge': [
                                     # Allow all other objects to be touched as well
                                     # (since hand will go through them in simulation) TODO desired behavior?
                                     AllowedCollision(type=AllowedCollision.BOUNDING_BOX, box_id=obj_idx,
                                                      terminating=False, required=obj_idx == current_object_idx)
                                     for obj_idx in range(0, len(objects))
                                 ]
                                 + [
                                     AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='tabletop',
                                                      terminating=False),
                                 ],

            }

        else:
            # TODO implement other strategies
            raise ValueError("Kinematics checks are currently only supported for surface grasps and wall grasps, "
                             "but strategy was " + strategy)

        # initialize stored trajectories for the given object
        self.stored_trajectories[(current_object_idx, current_ec_index)] = {}

        # Try to create the table from edges. As fall back use SurfaceGrasp frame (which might have a wrong orientation)
        # set this to False either way for disney use case
        table_from_edges = False
        if len(edges) < 2:

                # we use the SurfaceGrasp frame as a backup
                table_from_edges = False

        #        if table_frame is None:
        #            # we did not already compute a table pose backup...
        #            # Find potential SurfaceGrasp frames that can be used as a backup for table frames
        #            pot_table_frames = [e.transform for e in all_ecs if e.label == 'SurfaceGrasp']

        #            if len(pot_table_frames) > 0:
        #                table_frame = pot_table_frames[0]

        #                # Since the surface grasp frame is at the object center we have to translate it in z direction
        #                table_frame[2, 3] = table_frame[2, 3] - object['bounding_box'].z / 2.0

                        # Convert table frame to pose message
        #                table_pose = multi_object_params.transform_to_pose_msg(table_frame)

        #            else:
        #                # TODO limitation of the feasibility checker which needs at least two edges to create a table
        #                raise ValueError("Planning for {}, but not enough edges/table frames found".format(strategy))

        if table_frame is not None:
            print("TABLE POSE", table_pose)

        # The bounding boxes of all objects in message format
        bounding_boxes = []
        for obj in objects:
            obj_pose = multi_object_params.transform_to_pose_msg(obj['frame'])
            obj_bbox = SolidPrimitive(type=SolidPrimitive.BOX,
                                      dimensions=[obj['bounding_box'].x, obj['bounding_box'].y, obj['bounding_box'].z])

            bounding_boxes.append(BoundingBoxWithPose(box=obj_bbox, pose=obj_pose))
        print("BOUNDING_BOXES", bounding_boxes)

        all_steps_okay = True

        # perform the actual checks
        for idx, (motion, curr_goal) in enumerate(zip(checked_motions, goals)):

            manifold_name = motion + '_manifold'

            goal_pose = multi_object_params.transform_to_pose_msg(curr_goal)
            print("GOAL_POSE", goal_pose)
            print("INIT_CONF", curr_start_config)

            goal_manifold_frame = multi_object_params.transform_to_pose_msg(goal_manifold_frames[motion])  # TODO change dictionaries to Motion class?

            goal_manifold_orientation = geometry_msgs.msg.Quaternion(x=goal_manifold_orientations[motion][0],
                                                                     y=goal_manifold_orientations[motion][1],
                                                                     z=goal_manifold_orientations[motion][2],
                                                                     w=goal_manifold_orientations[motion][3])

            # TODO: used for debugging purposes, remove at some point
            static_transformStamped = geometry_msgs.msg.TransformStamped()

            static_transformStamped.header.stamp = rospy.Time.now()
            static_transformStamped.header.frame_id = "world"
            static_transformStamped.child_frame_id = "ec_{}_goal_{}".format(current_ec_index, idx)

            static_transformStamped.transform.translation.x = goal_pose.position.x
            static_transformStamped.transform.translation.y = goal_pose.position.y
            static_transformStamped.transform.translation.z = goal_pose.position.z

            static_transformStamped.transform.rotation.x = goal_manifold_orientation.x
            static_transformStamped.transform.rotation.y = goal_manifold_orientation.y
            static_transformStamped.transform.rotation.z = goal_manifold_orientation.z
            static_transformStamped.transform.rotation.w = goal_manifold_orientation.w

            tf_broadcaster.sendTransform(static_transformStamped)

            # Ocado use case:
            # check_feasibility = rospy.ServiceProxy('/check_kinematics', kin_check_srv.CheckKinematics)

            # Disney use case:
            check_feasibility = rospy.ServiceProxy('/check_kinematics_tabletop', kin_check_srv.CheckKinematicsTabletop)

            print("allowed", allowed_collisions[motion])

            print("Call check kinematics for " + motion + " (" + strategy + ")\nGoal:\n" + str(curr_goal))

            res = check_feasibility(initial_configuration=curr_start_config,
                                    goal_pose=goal_pose,
                                    table_surface_pose=table_pose,
                                    bounding_boxes_with_poses=bounding_boxes,
                                    goal_manifold_frame=goal_manifold_frame,
                                    min_position_deltas=params[manifold_name]['min_position_deltas'],
                                    max_position_deltas=params[manifold_name]['max_position_deltas'],
                                    goal_manifold_orientation=goal_manifold_orientation,
                                    min_orientation_deltas=params[manifold_name]['min_orientation_deltas'],
                                    max_orientation_deltas=params[manifold_name]['max_orientation_deltas'],
                                    allowed_collisions=allowed_collisions[motion],
                                    edge_frames=edges,
                                    table_from_edges=table_from_edges
                                    )

            # try again with more lenient conditions if strategy is edge grasp
            if res.status == CheckKinematicsTabletopResponse.FAILED and strategy == "EdgeGrasp":
                res = check_feasibility(initial_configuration=curr_start_config,
                                        goal_pose=goal_pose,
                                        table_surface_pose=table_pose,
                                        bounding_boxes_with_poses=bounding_boxes,
                                        goal_manifold_frame=goal_manifold_frame,
                                        min_position_deltas=params[manifold_name]['min_position_deltas'],
                                        max_position_deltas=params[manifold_name]['max_position_deltas'],
                                        goal_manifold_orientation=goal_manifold_orientation,
                                        min_orientation_deltas=[x*2. for x in params[manifold_name]['min_orientation_deltas']],
                                        max_orientation_deltas=[x*2. for x in params[manifold_name]['max_orientation_deltas']],
                                        allowed_collisions=allowed_collisions[motion],
                                        edge_frames=edges,
                                        table_from_edges=table_from_edges
                                        )

            print("check feasibility result was: " + str(res.status))

            if res.status == CheckKinematicsTabletopResponse.FAILED:
                # trajectory is not feasible and no alternative was found, directly return 0
                return 0

            elif res.status == CheckKinematicsTabletopResponse.REACHED_SAMPLED:
                # original trajectory is not feasible, but alternative was found => save it
                self.stored_trajectories[(current_object_idx, current_ec_index)][motion] = AlternativeBehavior(res, curr_start_config)
                curr_start_config = res.final_configuration
                all_steps_okay = False
                print("FOUND ALTERNATIVE. New Start: ", curr_start_config)

            elif res.status == CheckKinematicsTabletopResponse.REACHED_INITIAL:
                # original trajectory is feasible, we save the alternative in case a later motion is not possible.
                self.stored_trajectories[(current_object_idx, current_ec_index)][motion] = AlternativeBehavior(res, curr_start_config)
                curr_start_config = res.final_configuration
                print("USE NORMAL. Start: ", curr_start_config)

            else:
                raise ValueError(
                    "check_kinematics: No handler for result status of {} implemented".format(res.status))

        if all_steps_okay:
            # if all steps are okay use original trajectory TODO only replace preceding steps!

            ## we also want to execute trajectories in this case, therefore commenting next line
            # self.stored_trajectories[(current_object_idx, current_ec_index)] = {}
            pass

        #return self.pdf_object_strategy(object_params) * self.pdf_object_ec(object_params, ec_frame,strategy)
        return 1

    def black_list_risk_regions(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames,
                                ifco_in_base_transform):

        object = objects[current_object_idx]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        zone_fac = self.black_list_unreachable_zones(object, object_params, ifco_in_base_transform, strategy)
        wall_fac = self.black_list_walls(current_ec_index, all_ec_frames, strategy)

        return 1.0 #zone_fac * wall_fac TODO re-insert?

    @staticmethod
    def transform_to_pose(in_transform):
        # convert 4x4 matrix to trans + rot
        scale, shear, angles, translation, persp = tra.decompose_matrix(in_transform)
        orientation_quat = tra.quaternion_from_euler(angles[0], angles[1], angles[2])
        return translation, orientation_quat

    @staticmethod
    def transform_to_pose_msg(in_transform):
        trans, rot = multi_object_params.transform_to_pose(in_transform)
        trans = geometry_msgs.msg.Point(x=trans[0], y=trans[1], z=trans[2])
        rot = geometry_msgs.msg.Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        return geometry_msgs.msg.Pose(position=trans, orientation=rot)

    def reset_kinematic_checks_information(self):
        self.stored_trajectories = {}

    # ------------------------------------------------------------- #
    # object-environment-hand based heuristic, q_value for grasping
    def heuristic(self, current_object_idx, objects, current_ec_index, all_ecs, ifco_in_base_transform, handarm_params):

        object = objects[current_object_idx]
        strategy = all_ecs[current_ec_index].label

        ec_frame = all_ecs[current_ec_index].transform
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        feasibility_checker = rospy.get_param("feasibility_check/active", default="TUB")
        if feasibility_checker == 'TUB':
            feasibility_fun = partial(self.heuristics_and_check_kinematic_feasibility, current_ec_index, all_ecs, strategy, ifco_in_base_transform,
                                                   current_object_idx, objects, handarm_params)

        elif feasibility_checker == 'Ocado':
            # TODO integrate ocado
            raise ValueError("Ocado feasibility checker not integrated yet!")

        else:
            # Since we are in disney use case, only black list wall, but not regions.
            feasibility_fun = partial(self.black_list_edges, current_ec_index, all_ecs, strategy, ifco_in_base_transform,
                                      objects, current_object_idx, handarm_params)

        print("---------------------")
        print(self.pdf_object_strategy(object_params))
        print(self.pdf_object_ec(object_params, ec_frame, strategy))
        f = feasibility_fun()
        print(f)

        print("---------------------")

        q_val = 1
        q_val = q_val * \
            self.pdf_object_strategy(object_params) * \
            self.pdf_object_ec(object_params, ec_frame, strategy) * \
            f#feasibility_fun()

        print("qval", q_val)
        #print(" ** q_val = {} blaklisted={}".format(q_val, self.black_list_walls(current_ec_index, all_ec_frames)))
        return q_val

## --------------------------------------------------------- ##
    # find the max probability and if there are more than one return one randomly
    def argmax_h(self, Q_matrix):
        # find max probablity in list

        indeces_of_max = np.argwhere(Q_matrix == Q_matrix.max())
        print("indeces_of_max  = {}".format(indeces_of_max ))

        print ("Q={}".format(Q_matrix))
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
                return -1, -1

            all_ecs = []
            for j, ec in enumerate(ecs):
                all_ecs.append(
                    EnvironmentalConstraint(graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform)),
                                            ec.label))
                print("ecs:{}".format(graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform))))

            for curr_ec_idx, ec in enumerate(ecs):
                # the ec frame must be in the same reference frame as the object
                ec_frame_in_base = graph_in_base.dot(self.transform_msg_to_homogenous_tf(ec.transform))
                Q_matrix[i,curr_ec_idx] = self.heuristic(i, objects, curr_ec_idx, all_ecs, ifco_in_base_transform,
                                               handarm_parameters)

        # print (" ** h_mx = {}".format(Q_matrix))

        if Q_matrix.max() == 0.0:
            rospy.logwarn("No Suitable Grasp Found - PLEASE REPLAN!!!")
            if rospy.get_param("feasibility_check/active", default=True):
                # If not even the feasibilty checker could find an alternative signal a planner failure
                return -1, -1


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
