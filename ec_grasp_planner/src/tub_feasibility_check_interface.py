import math
import rospy
import numpy as np
from tf import transformations as tra

from geometry_graph_msgs.msg import Node, geometry_msgs
from tub_feasibility_check import srv as kin_check_srv
from tub_feasibility_check.msg import BoundingBoxWithPose, AllowedCollision
from tub_feasibility_check.srv import CheckKinematicsResponse
from shape_msgs.msg import SolidPrimitive

import GraspFrameRecipes
import planner_utils as pu


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


class FeasibilityQueryParameters:

    def __init__(self, checked_motions, goals, allowed_collisions, goal_manifold_frames, goal_manifold_orientations):
        # TODO change multiple dictionaries to one Motion class?

        # This list includes the checked motions in order (They have to be sequential!)
        self.checked_motions = checked_motions
        # The goal poses of the respective motions in op-space (index has to match index of checked_motions)
        self.goals = goals
        # The collisions that are allowed in message format per motion
        self.allowed_collisions = allowed_collisions
        # TODO docu
        self.goal_manifold_frames = goal_manifold_frames
        # TODO docu
        self.goal_manifold_orientations = goal_manifold_orientations


def get_matching_ifco_wall(ifco_in_base_transform, ec_frame):
    # transforms points in base frame to ifco frame
    base_in_ifco_transform = tra.inverse_matrix(ifco_in_base_transform)

    # ec x axis in ifco frame
    ec_x_axis = base_in_ifco_transform.dot(ec_frame)[0:3, 0]
    ec_z_axis = base_in_ifco_transform.dot(ec_frame)[0:3, 2]

    # we can't check for zero because of small errors in the frame (due to vision or numerical uncertainty)
    space_thresh = 0.1

    # one could also check for dot-product = 0 instead of using the x-axis but this is prone to the same errors
    if ec_z_axis.dot(np.array([1, 0, 0])) > space_thresh and ec_x_axis.dot(np.array([0, 1, 0])) > space_thresh:
        # print("GET MATCHING=SOUTH", tf_dbg_call_to_string(ec_frame, frame_name='ifco_south'))
        return 'south'
    elif ec_z_axis.dot(np.array([1, 0, 0])) < -space_thresh and ec_x_axis.dot(np.array([0, 1, 0])) < -space_thresh:
        # print("GET MATCHING=NORTH", tf_dbg_call_to_string(ec_frame, frame_name='ifco_north'))
        return 'north'
    elif ec_z_axis.dot(np.array([0, 1, 0])) < -space_thresh and ec_x_axis.dot(np.array([1, 0, 0])) > space_thresh:
        # print("GET MATCHING=WEST", tf_dbg_call_to_string(ec_frame, frame_name='ifco_west'))
        return 'west'
    elif ec_z_axis.dot(np.array([0, 1, 0])) > space_thresh and ec_x_axis.dot(np.array([1, 0, 0])) < -space_thresh:
        # print("GET MATCHING=EAST", tf_dbg_call_to_string(ec_frame, frame_name='ifco_east'))
        return 'east'
    else:
        # This should never be reached. Just here to prevent bugs
        raise ValueError("ERROR: Could not identify matching ifco wall. Check frames!")


def get_matching_ifco_corner(ifco_in_base_transform, ec_frame):

    # transforms points in base frame to ifco frame
    base_in_ifco_transform = tra.inverse_matrix(ifco_in_base_transform)

    # ec (corner) z-axis in ifco frame
    ec_z_axis = base_in_ifco_transform.dot(ec_frame)[0:3, 2]

    # we can't check for zero because of small errors in the frame (due to vision or numerical uncertainty)
    space_thresh = 0.0  # 0.1

    if ec_z_axis.dot(np.array([1, 0, 0])) > space_thresh and ec_z_axis.dot(np.array([0, 1, 0])) > space_thresh:
        print("GET MATCHING=SOUTH_EAST", pu.tf_dbg_call_to_string(ec_frame, frame_name='ifco_southeast'))
        return 'south', 'east'
    elif ec_z_axis.dot(np.array([1, 0, 0])) > space_thresh and ec_z_axis.dot(np.array([0, 1, 0])) < -space_thresh:
        print("GET MATCHING=SOUTH_WEST", pu.tf_dbg_call_to_string(ec_frame, frame_name='ifco_southwest'))
        return 'south', 'west'
    elif ec_z_axis.dot(np.array([1, 0, 0])) < -space_thresh and ec_z_axis.dot(np.array([0, 1, 0])) < -space_thresh:
        print("GET MATCHING=NORTH_WEST", pu.tf_dbg_call_to_string(ec_frame, frame_name='ifco_northwest'))
        return 'north', 'west'
    elif ec_z_axis.dot(np.array([1, 0, 0])) < -space_thresh and ec_z_axis.dot(np.array([0, 1, 0])) > space_thresh:
        print("GET MATCHING=NORTH_EAST", pu.tf_dbg_call_to_string(ec_frame, frame_name='ifco_northeast'))
        return 'north', 'east'
    else:
        # This should never be reached. Just here to prevent bugs
        raise ValueError("ERROR: Could not identify matching ifco wall. Check frames!")


# Checks if the Y-Axis of the ifco frame points towards the robot (origin of base frame)
# The base frame is assumed to be the following way:
#   x points to the robots front
#   y points to the robots left (if you are behind the robot)
#   z points upwards
def ifco_transform_needs_to_be_flipped(ifco_in_base_transform):

    # we can't check for zero because of small errors in the frame (due to vision or numerical uncertainty)
    space_thresh = 0.05

    x_of_yaxis = ifco_in_base_transform[0, 1]
    x_of_translation = ifco_in_base_transform[0, 3]

    print(ifco_in_base_transform)
    print(space_thresh, x_of_yaxis, x_of_translation)

    if x_of_translation > space_thresh:
        # ifco is in front of robot
        return x_of_yaxis > 0
    elif x_of_translation < space_thresh:
        # ifco is behind the robot
        return x_of_yaxis < 0
    else:
        y_of_translation = ifco_in_base_transform[1, 3]
        y_of_yaxis = ifco_in_base_transform[1, 1]
        if y_of_translation < 0:
            # ifco is to the right of the robot
            return y_of_yaxis < 0
        else:
            # ifco is to the left of the robot
            return y_of_yaxis > 0


# This function will call TUB's feasibility checker to check a motion.
# If the motion is not feasible it will try to generate an alternative joint trajectory and place it into
# the given stored_trajectories argument (dictionary).
def check_kinematic_feasibility(current_object_idx, objects, object_params, current_ec_index, strategy, all_ec_frames,
                                ifco_in_base_transform, handarm_params, stored_trajectories):

    if handarm_params is None:
        raise ValueError("HandArmParameters can't be None, check callstack!")

    print("IFCO_BEFORE", pu.tf_dbg_call_to_string(ifco_in_base_transform, frame_name='ifco_before'))
    if ifco_transform_needs_to_be_flipped(ifco_in_base_transform):
        # flip the ifco transform such that it fulfills the requirements of the feasibilty checker
        # (y-axis of ifco points towards the robot)
        rospy.loginfo("Flip ifco transform for tub feasibilty checker")
        zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
        ifco_in_base_transform = ifco_in_base_transform.dot(zflip_transform)

    print("IFCO_AFTER", pu.tf_dbg_call_to_string(ifco_in_base_transform, frame_name='ifco_after'))

    object = objects[current_object_idx]
    ec_frame = all_ec_frames[current_ec_index]

    if object['type'] in handarm_params[strategy]:
        params = handarm_params[strategy][object['type']]
    else:
        params = handarm_params[strategy]['object']

    # The initial joint configuration (goToView config)
    # curr_start_config = rospy.get_param('planner_gui/robot_view_position') # TODO use current joint state instead?
    # TODO also check gotToView -> params['initial_goal'] (requires forward kinematics, or change to op-space)
    curr_start_config = params['initial_goal']

    if strategy == 'SurfaceGrasp':

        call_params = prepare_surface_grasp_parameter(objects, current_object_idx, object_params, params)

    elif strategy == "WallGrasp":

        selected_wall_name = get_matching_ifco_wall(ifco_in_base_transform, ec_frame)
        print("FOUND_EC: ", selected_wall_name)

        blocked_ecs = ['north', 'east', 'west']  # TODO move to config file?
        if selected_wall_name in blocked_ecs:
            rospy.loginfo("Skipped wall " + selected_wall_name + " (Blacklisted)")
            return 0

        call_params = prepare_wall_grasp_parameter(ec_frame, selected_wall_name, objects, current_object_idx,
                                                   object_params, ifco_in_base_transform, params)

    elif strategy == "CornerGrasp":

        selected_wall_names = get_matching_ifco_corner(ifco_in_base_transform, ec_frame)
        print("FOUND_EC: ", selected_wall_names)

        blocked_ecs = [('north', 'east'), ('north', 'west'), ('south', 'west')]  # TODO move to config file?
        if selected_wall_names in blocked_ecs:
            rospy.loginfo("Skipped corner " + selected_wall_names[0] + selected_wall_names[1] + " (Blacklisted)")
            return 0

        call_params = prepare_corner_grasp_parameter(ec_frame, selected_wall_names, objects, current_object_idx,
                                                     object_params, ifco_in_base_transform, params)

    else:
        raise ValueError("Kinematics checks are currently only supported for surface grasps and wall grasps, "
                         "but strategy was " + strategy)

    # initialize stored trajectories for the given object
    stored_trajectories[(current_object_idx, current_ec_index)] = {}

    # The pose of the ifco (in base frame) in message format
    ifco_pose = pu.transform_to_pose_msg(ifco_in_base_transform)
    print("IFCO_POSE", ifco_pose)

    # The bounding boxes of all objects in message format
    bounding_boxes = []
    for obj in objects:
        obj_pose = pu.transform_to_pose_msg(obj['frame'])
        obj_bbox = SolidPrimitive(type=SolidPrimitive.BOX,
                                  dimensions=[obj['bounding_box'].x, obj['bounding_box'].y, obj['bounding_box'].z])

        bounding_boxes.append(BoundingBoxWithPose(box=obj_bbox, pose=obj_pose))
    print("BOUNDING_BOXES", bounding_boxes)

    all_steps_okay = True

    # perform the actual checks
    for motion, curr_goal in zip(call_params.checked_motions, call_params.goals):

        manifold_name = motion + '_manifold'

        goal_pose = pu.transform_to_pose_msg(curr_goal)
        print("GOAL_POSE", goal_pose)
        print("INIT_CONF", curr_start_config)

        goal_manifold_frame = pu.transform_to_pose_msg(call_params.goal_manifold_frames[motion])

        goal_manifold_orientation = geometry_msgs.msg.Quaternion(x=call_params.goal_manifold_orientations[motion][0],
                                                                 y=call_params.goal_manifold_orientations[motion][1],
                                                                 z=call_params.goal_manifold_orientations[motion][2],
                                                                 w=call_params.goal_manifold_orientations[motion][3])

        check_feasibility = rospy.ServiceProxy('/check_kinematics', kin_check_srv.CheckKinematics)

        print("allowed", call_params.allowed_collisions[motion])

        print("Call check kinematics for " + motion + " (" + strategy + ")\nGoal:\n" + str(curr_goal))

        res = check_feasibility(initial_configuration=curr_start_config,
                                goal_pose=goal_pose,
                                ifco_pose=ifco_pose,
                                bounding_boxes_with_poses=bounding_boxes,
                                goal_manifold_frame=goal_manifold_frame,
                                min_position_deltas=params[manifold_name]['min_position_deltas'],
                                max_position_deltas=params[manifold_name]['max_position_deltas'],
                                goal_manifold_orientation=goal_manifold_orientation,
                                min_orientation_deltas=params[manifold_name]['min_orientation_deltas'],
                                max_orientation_deltas=params[manifold_name]['max_orientation_deltas'],
                                allowed_collisions=call_params.allowed_collisions[motion]
                                )

        print("check feasibility result was: " + str(res.status))

        if res.status == CheckKinematicsResponse.FAILED:
            # trajectory is not feasible and no alternative was found, directly return 0
            return 0

        elif res.status == CheckKinematicsResponse.REACHED_SAMPLED:
            # original trajectory is not feasible, but alternative was found => save it
            stored_trajectories[(current_object_idx, current_ec_index)][motion] = AlternativeBehavior(res, curr_start_config)
            curr_start_config = res.final_configuration
            all_steps_okay = False
            print("FOUND ALTERNATIVE. New Start: ", curr_start_config)

        elif res.status == CheckKinematicsResponse.REACHED_INITIAL:
            # original trajectory is feasible, we save the alternative in case a later motion is not possible.
            stored_trajectories[(current_object_idx, current_ec_index)][motion] = AlternativeBehavior(res, curr_start_config)
            curr_start_config = res.final_configuration
            print("USE NORMAL. Start: ", curr_start_config)

        else:
            raise ValueError(
                "check_kinematics: No handler for result status of {} implemented".format(res.status))

    if all_steps_okay:
        # if all steps are okay use original trajectory TODO only replace preceding steps!
        stored_trajectories[(current_object_idx, current_ec_index)] = {}
        pass

    # Either the initial trajectory was possible or an alternative behavior was generated
    return 1.0


def prepare_surface_grasp_parameter(objects, current_object_idx, object_params, params):
    # use kinematic checks
    # TODO create proxy; make it a persistent connection?

    # Code duplication from planner.py TODO put at a shared location

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
    pre_grasp_pose = goal_.dot(params['pre_approach_transform'])

    # down_dist = params['down_dist']  #  dist lower than ifco bottom: behavior of the high level planner
    # dist = z difference to object centroid (both transformations are w.r.t. to world frame
    # (more realistic behavior since we have to touch the object for a successful grasp)
    down_dist = pre_grasp_pose[2, 3] - object_params['frame'][2, 3]  # get z-translation difference

    # goal pose for go down movement
    go_down_pose = tra.translation_matrix([0, 0, -down_dist]).dot(pre_grasp_pose)

    post_grasp_pose = params['post_grasp_transform'].dot(
        go_down_pose)  # TODO it would be better to allow relative motion as goal frames

    checked_motions = ["pre_approach",
                       "go_down"]  # , "post_grasp_rot"] ,go_up, go_drop_off  # TODO what about remaining motions? (see wallgrasp)

    goals = [pre_grasp_pose, go_down_pose]  # , post_grasp_pose]

    # TODO what about using the bounding boxes as for automatic goal manifold calculation?

    # Take orientation of object but translation of pre grasp pose
    pre_grasp_pos_manifold = np.copy(object_params['frame'])
    pre_grasp_pos_manifold[:3, 3] = tra.translation_from_matrix(pre_grasp_pose)

    goal_manifold_frames = {
        'pre_approach': pre_grasp_pos_manifold,

        # Use object frame for resampling
        'go_down': np.copy(object_params['frame'])  # TODO change that again to go_down_pose!?
    }

    goal_manifold_orientations = {
        # use hand orientation
        'pre_approach': tra.quaternion_from_matrix(pre_grasp_pose),

        # Use object orientation
        'go_down': tra.quaternion_from_matrix(go_down_pose),
        # tra.quaternion_from_matrix(object_params['frame'])  # TODO use hand orietation instead?
    }

    # The collisions that are allowed per motion in message format
    allowed_collisions = {
        # no collisions are allowed during going to pre_grasp pose
        'pre_approach': [],

        'go_down': [AllowedCollision(type=AllowedCollision.BOUNDING_BOX, box_id=current_object_idx,
                                     terminating=True, required=True),
                    AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                     constraint_name='bottom', terminating=False)] +

                   [AllowedCollision(type=AllowedCollision.BOUNDING_BOX, box_id=obj_idx, terminating=False)
                    for obj_idx, o in enumerate(objects) if obj_idx != current_object_idx and
                    params['go_down_allow_touching_other_objects']
                    ],

        # TODO also account for the additional object in a way?
        'post_grasp_rot': [AllowedCollision(type=AllowedCollision.BOUNDING_BOX, box_id=current_object_idx,
                                            terminating=True),
                           AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                            constraint_name='bottom', terminating=False)]
    }

    print("ALLOWED COLLISIONS:", allowed_collisions)

    return FeasibilityQueryParameters(checked_motions, goals, allowed_collisions, goal_manifold_frames,
                                      goal_manifold_orientations)


def prepare_wall_grasp_parameter(ec_frame, selected_wall_name, objects, current_object_idx, object_params,
                                 ifco_in_base_transform, params):

    # hand pose above and behind the object
    pre_approach_transform = params['pre_approach_transform']

    wall_frame = np.copy(ec_frame)
    wall_frame[:3, 3] = tra.translation_from_matrix(object_params['frame'])
    # apply hand transformation
    ec_hand_frame = wall_frame.dot(params['hand_transform'])

    # ec_hand_frame = (ec_frame.dot(params['hand_transform']))
    pre_approach_pose = ec_hand_frame.dot(pre_approach_transform)

    # down_dist = params['down_dist']  #  dist lower than ifco bottom: behavior of the high level planner
    # dist = z difference to ifco bottom minus hand frame offset (dist from hand frame to collision point)
    # (more realistic behavior since we have a force threshold when going down to the bottom)
    bounded_down_dist = pre_approach_pose[2, 3] - ifco_in_base_transform[2, 3]
    hand_frame_to_bottom_offset = 0.07  # 7cm TODO maybe move to handarm_parameters.py
    bounded_down_dist = min(params['down_dist'], bounded_down_dist - hand_frame_to_bottom_offset)

    # goal pose for go down movement
    go_down_pose = tra.translation_matrix([0, 0, -bounded_down_dist]).dot(pre_approach_pose)

    # pose after lifting. This is somewhat fake, since the real go_down_pose will be determined by
    # the FT-Switch during go_down and the actual lifted distance by the TimeSwitch (or a pose switch in case
    # the robot allows precise small movements) TODO better solution?
    fake_lift_up_dist = np.min([params['lift_dist'], 0.01])  # 1cm
    corrective_lift_pose = tra.translation_matrix([0, 0, fake_lift_up_dist]).dot(go_down_pose)

    dir_wall = tra.translation_matrix([0, 0, -params['sliding_dist']])
    # TODO sliding_distance should be computed from wall and hand frame.
    # slide direction is given by the normal of the wall
    wall_frame = np.copy(ec_frame)
    dir_wall[:3, 3] = wall_frame[:3, :3].dot(dir_wall[:3, 3])

    # normal goal pose behind the wall
    slide_to_wall_pose = dir_wall.dot(corrective_lift_pose)

    # now project it into the wall plane!
    z_projection = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]])

    to_wall_plane_transform = wall_frame.dot(z_projection.dot(tra.inverse_matrix(wall_frame).dot(slide_to_wall_pose)))
    slide_to_wall_pose[:3, 3] = tra.translation_from_matrix(to_wall_plane_transform)

    # TODO remove code duplication with planner.py (refactor code snippets to function calls) !!!!!!!

    checked_motions = ['pre_approach', 'go_down', 'corrective_lift',
                       'slide_to_wall']  # TODO overcome problem of FT-Switch after go_down

    goals = [pre_approach_pose, go_down_pose, corrective_lift_pose, slide_to_wall_pose]  # TODO see checked_motions

    # Take orientation of object but translation of pre grasp pose
    pre_grasp_pos_manifold = np.copy(object_params['frame'])
    pre_grasp_pos_manifold[:3, 3] = tra.translation_from_matrix(pre_approach_pose)

    slide_pos_manifold = np.copy(slide_to_wall_pose)

    goal_manifold_frames = {
        'pre_approach': pre_grasp_pos_manifold,

        # Use object frame for sampling
        'go_down': np.copy(go_down_pose),

        'corrective_lift': np.copy(corrective_lift_pose),
    # should always be the same frame as go_down # TODO use world orientation?

        # Use wall frame for sampling. Keep in mind that the wall frame has different orientation, than world.
        'slide_to_wall': slide_pos_manifold,
    }

    goal_manifold_orientations = {
        # use hand orientation
        'pre_approach': tra.quaternion_from_matrix(pre_approach_pose),

        # Use object orientation
        'go_down': tra.quaternion_from_matrix(go_down_pose),  # TODO use hand orietation instead?

        # should always be the same orientation as go_down
        'corrective_lift': tra.quaternion_from_matrix(corrective_lift_pose),

        # use wall orientation
        'slide_to_wall': tra.quaternion_from_matrix(wall_frame),
    }

    allowed_collisions = {

        # 'init_joint': [],

        # no collisions are allowed during going to pre_grasp pose
        'pre_approach': [],

        # Only allow touching the bottom of the ifco
        'go_down': [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                     terminating=False),
                    ],

        'corrective_lift': [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                                  terminating=False),
                      ],

        # TODO also allow all other obejcts to be touched during sliding motion
        'slide_to_wall': [
                             # Allow all other objects to be touched as well
                             # (since hand will go through them in simulation) TODO desired behavior?
                             AllowedCollision(type=AllowedCollision.BOUNDING_BOX, box_id=obj_idx,
                                              terminating=False, required=obj_idx == current_object_idx)
                             for obj_idx in range(0, len(objects))
                         ] + [
                             AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                              constraint_name=selected_wall_name, terminating=False),

                             AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                              terminating=False),
                         ],

    }

    return FeasibilityQueryParameters(checked_motions, goals, allowed_collisions, goal_manifold_frames,
                                      goal_manifold_orientations)


def prepare_corner_grasp_parameter(ec_frame, selected_wall_names, objects, current_object_idx, object_params,
                                   ifco_in_base_transform, params):

    rospy.logerr("Kinematics checks are currently only supported for surface grasps and wall grasps, "
                 "but strategy was CornerGrasp")

    # hand pose above and behind the object
    pre_approach_transform = params['pre_approach_transform']

    corner_frame = np.copy(ec_frame)
    print("Prepare Corner: ", pu.tf_dbg_call_to_string(corner_frame, "prepare"))
    used_ec_frame, corner_frame_alpha_zero = GraspFrameRecipes.get_derived_corner_grasp_frames(corner_frame,
                                                                                               object_params['frame'])
    pre_approach_pose = used_ec_frame.dot(params['hand_transform'].dot(pre_approach_transform)) # TODO order of hand and pre_approach

    # down_dist = params['down_dist']  #  dist lower than ifco bottom: behavior of the high level planner
    # dist = z difference to ifco bottom minus hand frame offset (dist from hand frame to collision point)
    # (more realistic behavior since we have a force threshold when going down to the bottom)
    bounded_down_dist = pre_approach_pose[2, 3] - ifco_in_base_transform[2, 3]
    hand_frame_to_bottom_offset = 0.07  # 7cm TODO maybe move to handarm_parameters.py
    bounded_down_dist = min(params['down_dist'], bounded_down_dist - hand_frame_to_bottom_offset)

    # goal pose for go down movement
    go_down_pose = tra.translation_matrix([0, 0, -bounded_down_dist]).dot(pre_approach_pose)

    # pose after lifting. This is somewhat fake, since the real go_down_pose will be determined by
    # the FT-Switch during go_down and the actual lifted distance by the TimeSwitch (or a pose switch in case
    # the robot allows precise small movements) TODO better solution?
    fake_lift_up_dist = np.min([params['lift_dist'], 0.01])  # 1cm
    corrective_lift_pose = tra.translation_matrix([0, 0, fake_lift_up_dist]).dot(go_down_pose)

    sliding_dist = params['sliding_dist']
    wall_dir = tra.translation_matrix([0, 0, -sliding_dist])
    # slide direction is given by the corner_frame_alpha_zero
    wall_dir[:3, 3] = corner_frame_alpha_zero[:3, :3].dot(wall_dir[:3, 3])

    slide_to_wall_pose = wall_dir.dot(corrective_lift_pose)

    # now project it into the wall plane!
    z_projection = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]])

    to_wall_plane_transform = corner_frame_alpha_zero.dot(z_projection.dot(tra.inverse_matrix(corner_frame_alpha_zero).dot(slide_to_wall_pose)))
    slide_to_wall_pose[:3, 3] = tra.translation_from_matrix(to_wall_plane_transform)

    checked_motions = ['pre_approach', 'go_down', 'corrective_lift', 'slide_to_wall']

    goals = [pre_approach_pose, go_down_pose, corrective_lift_pose, slide_to_wall_pose]

    # Take orientation of object but translation of pre grasp pose
    pre_grasp_pos_manifold = np.copy(object_params['frame'])
    pre_grasp_pos_manifold[:3, 3] = tra.translation_from_matrix(pre_approach_pose)

    slide_pos_manifold = np.copy(slide_to_wall_pose)

    goal_manifold_frames = {
        'pre_approach': pre_grasp_pos_manifold,

        # Use object frame for sampling
        'go_down': np.copy(go_down_pose),

        'corrective_lift': np.copy(corrective_lift_pose),
    # should always be the same frame as go_down # TODO use world orientation?

        # Use wall frame for sampling. Keep in mind that the wall frame has different orientation, than world.
        'slide_to_wall': slide_pos_manifold,
    }

    goal_manifold_orientations = {
        # use hand orientation
        'pre_approach': tra.quaternion_from_matrix(pre_approach_pose),

        # Use object orientation
        'go_down': tra.quaternion_from_matrix(go_down_pose),  # TODO use hand orietation instead?

        # should always be the same orientation as go_down
        'corrective_lift': tra.quaternion_from_matrix(corrective_lift_pose),

        # use wall orientation
        'slide_to_wall': tra.quaternion_from_matrix(corner_frame),  # TODO is that the right one?
    }

    allowed_collisions = {

        # 'init_joint': [],

        # no collisions are allowed during going to pre_grasp pose
        'pre_approach': [],

        # Only allow touching the bottom of the ifco
        'go_down': [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                     terminating=False),
                    ],

        'corrective_lift': [AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                                  terminating=False),
                      ],

        'slide_to_wall': [
                             # Allow all other objects to be touched as well
                             # (since hand will go through them in simulation)
                             AllowedCollision(type=AllowedCollision.BOUNDING_BOX, box_id=obj_idx,
                                              terminating=False, required=obj_idx == current_object_idx)
                             for obj_idx in range(0, len(objects))
                         ] + [
                             AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                              constraint_name=selected_wall_names[0], terminating=False),

                             AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT,
                                              constraint_name=selected_wall_names[1], terminating=False),

                             AllowedCollision(type=AllowedCollision.ENV_CONSTRAINT, constraint_name='bottom',
                                              terminating=False),
                         ],

    }

    return FeasibilityQueryParameters(checked_motions, goals, allowed_collisions, goal_manifold_frames,
                                      goal_manifold_orientations)

