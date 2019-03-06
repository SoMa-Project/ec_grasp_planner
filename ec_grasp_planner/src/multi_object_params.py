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

    # TODO move that to a separate file?
    def check_kinematic_feasibility(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames,
                                    ifco_in_base_transform, handarm_params):

        object = objects[current_object_idx]
        ec_frame = all_ec_frames[current_ec_index]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        # This list includes the checked motions in order (They have to be sequential!)
        checked_motions = []
        # The goal poses of the respective motions in op-space (index has to match index of checked_motions)
        goals = []
        # The collisions that are allowed in message format per motion
        allowed_collisions = {}

        # The initial joint configuration (goToView config)
        curr_start_config = [0.457929, 0.295013, -0.232804, 2.0226, 0.0, 1.50907, 1.0]  # TODO use line below!
        # curr_start_config = rospy.get_param('planner_gui/robot_view_position') # TODO use current joint state instead?

        # TODO maybe move the kinematic stuff to separate file

        if strategy == 'SurfaceGrasp':
            # use kinematic checks
            # TODO create proxy; make it a persistent connection?

            # Code duplication from planner.py TODO put at a shared location

            if object['type'] in handarm_params['surface_grasp']:
                params = handarm_params['surface_grasp'][object['type']]
            else:
                params = handarm_params['surface_grasp']['object']

            # TODO needed ----------------------
            x_flip_transform = tra.concatenate_matrices(
                tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(180.0), [1, 0, 0]))

            table_frame = ec_frame.dot(x_flip_transform)

            # Since the surface grasp frame is at the object center we have to translate it in z direction
            table_frame[2, 3] = table_frame[2, 3] - object['bounding_box'].z / 2.0

            #object_frame = object_params['frame'].dot(x_flip_transform)
            object_frame = object_params['frame']

            print("DBG_FRAME 1", multi_object_params.transform_to_pose_msg(object_frame))
            # TODO needed ----------------------

            goal_ = np.copy(object_frame)
            # we rotate the frame to align the hand with the long object axis and the table surface
            x_axis = object_frame[:3, 0]
            z_axis = table_frame[:3, 2]
            y_axis = np.cross(z_axis, x_axis)

            goal_[:3, 0] = x_axis
            goal_[:3, 1] = y_axis / np.linalg.norm(y_axis)
            goal_[:3, 2] = z_axis

            goal_ = goal_.dot(params['hand_transform'])

            print("DBG_FRAME 2",  multi_object_params.transform_to_pose_msg(goal_))

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
            pre_grasp_pose = goal_.dot(params['pregrasp_transform'])

            # down_dist = params['down_dist']  #  dist lower than ifco bottom: behavior of the high level planner
            # dist = z difference to object centroid (both transformations are w.r.t. to world frame
            # (more realistic behavior since we have to touch the object for a successful grasp)
            down_dist = pre_grasp_pose[2, 3] - object_params['frame'][2, 3]  # get z-translation difference

            # goal pose for go down movement
            go_down_pose = tra.translation_matrix([0, 0, -down_dist]).dot(pre_grasp_pose)

            post_grasp_pose = params['post_grasp_transform'].dot(go_down_pose)  # TODO it would be better to allow relative motion as goal frames

            checked_motions = ["pre_grasp", "go_down"]#, "post_grasp_rot"] ,go_up, go_drop_off  # TODO what about remaining motions? (see wallgrasp)

            goals = [pre_grasp_pose, go_down_pose]#, post_grasp_pose]

            # TODO what about using the bounding boxes as for automatic goal manifold calculation?

            # Take orientation of object but translation of pre grasp pose
            pre_grasp_pos_manifold = np.copy(object_params['frame'])
            pre_grasp_pos_manifold[:3, 3] = tra.translation_from_matrix(pre_grasp_pose)

            goal_manifold_frames = {
                'pre_grasp': pre_grasp_pos_manifold,

                # Use object frame for resampling
                'go_down': np.copy(object_params['frame'])  # TODO change that again to go_down_pose!?
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
                                             box_id=current_object_idx, terminating=True, required=True),
                            AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                             constraint_name='bottom', terminating=False)],

                # TODO also account for the additional object in a way?
                'post_grasp_rot': [AllowedCollision(type=AllowedCollision.BOUNDING_BOX,
                                                    box_id=current_object_idx, terminating=True),
                                   AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                                    constraint_name='bottom', terminating=False)]
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

            go_down_pose = tra.translation_matrix([0, 0, -bounded_lift_down_dist]).dot(pre_approach_pose)

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

        else:
            # TODO implement other strategies
            raise ValueError("Kinematics checks are currently only supported for surface grasps and wall grasps, "
                             "but strategy was " + strategy)
            #return 0

        # initialize stored trajectories for the given object
        self.stored_trajectories[(current_object_idx, current_ec_index)] = {}

        # The pose of the ifco (in base frame) in message format
        # ifco_pose = multi_object_params.transform_to_pose_msg(tra.inverse_matrix(ifco_in_base_transform))
        # TODO this only works for SurfaceGrasp. EdgeGrasp needs a different table representation.

        x_flip_transform = tra.concatenate_matrices(
            tra.translation_matrix([0, 0, 0]), tra.rotation_matrix(math.radians(180.0), [1, 0, 0]))

        if strategy != "SurfaceGrasp":
            rospy.logerr("This only works for surface grasp right now (since we have the table frame)")

        table_pose = multi_object_params.transform_to_pose_msg(table_frame)  # TODO This only works for surface grasp right now (since we have the table frame)
        #table_pose = multi_object_params.transform_to_pose_msg(np.copy(ec_frame).dot(x_flip_transform)) # TODO remove copy
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
        for motion, curr_goal in zip(checked_motions, goals):

            manifold_name = motion + '_manifold'

            goal_pose = multi_object_params.transform_to_pose_msg(curr_goal)
            print("GOAL_POSE", goal_pose)
            print("INIT_CONF", curr_start_config)

            goal_manifold_frame = multi_object_params.transform_to_pose_msg(goal_manifold_frames[motion])  # TODO change dictionaries to Motion class?

            goal_manifold_orientation = geometry_msgs.msg.Quaternion(x=goal_manifold_orientations[motion][0],
                                                                     y=goal_manifold_orientations[motion][1],
                                                                     z=goal_manifold_orientations[motion][2],
                                                                     w=goal_manifold_orientations[motion][3])

            # Ocado use case:
            # check_feasibility = rospy.ServiceProxy('/check_kinematics', kin_check_srv.CheckKinematics)

            # Disney use case:
            check_feasibility = rospy.ServiceProxy('/check_kinematics_tabletop', kin_check_srv.CheckKinematicsTabletop)

            print("allowed", allowed_collisions[motion])

            print("Call check kinematics for " + motion + " " + str(curr_goal))#Arguments: \n" + yaml.safe_dump(args))

            res = check_feasibility(initial_configuration=curr_start_config,
                                    goal_pose=goal_pose,
                                    table_surface_pose=table_pose,
                                    table_from_edges=False,
                                    bounding_boxes_with_poses=bounding_boxes,
                                    goal_manifold_frame=goal_manifold_frame,
                                    min_position_deltas=params[manifold_name]['min_position_deltas'],
                                    max_position_deltas=params[manifold_name]['max_position_deltas'],
                                    goal_manifold_orientation=goal_manifold_orientation,
                                    min_orientation_deltas=params[manifold_name]['min_orientation_deltas'],
                                    max_orientation_deltas=params[manifold_name]['max_orientation_deltas'],
                                    allowed_collisions=allowed_collisions[motion]
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
                # original trajectory is feasible, we don't have to save an alternative TODO update comment & code
                self.stored_trajectories[(current_object_idx, current_ec_index)][motion] = AlternativeBehavior(res, curr_start_config)
                curr_start_config = res.final_configuration
                print("USE NORMAL. Start: ", curr_start_config)

            else:
                raise ValueError(
                    "check_kinematics: No handler for result status of {} implemented".format(res.status))

        if all_steps_okay:
            # if all steps are okay use original trajectory TODO only replace preceding steps!
            # self.stored_trajectories[(current_object_idx, current_ec_index)] = {} # TODO add again
            pass

        return self.pdf_object_strategy(object_params) * self.pdf_object_ec(object_params, ec_frame,
                                                                            strategy)

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
    def heuristic(self, current_object_idx, objects, current_ec_index, strategy, all_ec_frames, ifco_in_base_transform, handarm_params):

        object = objects[current_object_idx]

        ec_frame = all_ec_frames[current_ec_index]
        object_params = self.data[object['type']][strategy]
        object_params['frame'] = object['frame']

        feasibility_checker = rospy.get_param("feasibility_check/active", default="TUB")
        if feasibility_checker == 'TUB':
            feasibility_fun = partial(self.check_kinematic_feasibility, current_object_idx, objects, current_ec_index,
                                      strategy, all_ec_frames, ifco_in_base_transform, handarm_params)

        elif feasibility_checker == 'Ocado':
            # TODO integrate ocado
            raise ValueError("Ocado feasibility checker not integrated yet!")

        else:
            # Since we are in disney use case, only black list wall, but not regions.
            feasibility_fun = partial(self.black_list_walls, current_ec_index, all_ec_frames, strategy)

        q_val = 1
        q_val = q_val * \
            self.pdf_object_strategy(object_params) * \
            self.pdf_object_ec(object_params, ec_frame, strategy) * \
            feasibility_fun()

        # print(" ** q_val = {} blaklisted={}".format(q_val, self.black_list_walls(current_ec_index, all_ec_frames)))
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
