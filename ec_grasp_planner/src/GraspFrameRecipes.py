import numpy as np
from tf import transformations as tra
from planner_utils import convert_transform_msg_to_homogeneous_tf


def get_derived_corner_grasp_frames(corner_frame, object_pose):

    ec_frame = np.copy(corner_frame)
    ec_frame[:3, 3] = tra.translation_from_matrix(object_pose)
    # y-axis stays the same, lets norm it just to go sure
    y = ec_frame[:3, 1] / np.linalg.norm(ec_frame[:3, 1])
    # z-axis is (roughly) the vector from corner to object
    z = ec_frame[:3, 3] - corner_frame[:3, 3]
    # z-axis should lie in the y-plane, so we subtract the part that is perpendicular to the y-plane
    z = z - (np.dot(z, y) * y)
    z = z / np.linalg.norm(z)
    # x-axis is perpendicular to y- and z-axis, again normed to go sure
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    # the rotation part is overwritten with the new axis
    # ec_frame[:3, :3] = np.stack((x, y, z), axis=1) # TODO this line requires a newer version of numpy
    ec_frame[:3, :3] = tra.inverse_matrix(np.vstack((x, y, z)))  # <- This one is the downward compatible version

    corner_frame_alpha_zero = np.copy(corner_frame)
    corner_frame_alpha_zero[:3, :3] = np.copy(ec_frame[:3, :3])

    return ec_frame, corner_frame_alpha_zero


def get_surface_pregrasp_pose_in_base_frame(pre_grasp_in_object_frame, object_pose):
    return object_pose.dot(pre_grasp_in_object_frame)


def get_wall_pregrasp_pose_in_base_frame(chosen_node, pre_grasp_in_object_frame, object_pose, graph_in_base):

    object_pos_with_ec_orientation = graph_in_base.dot(
        convert_transform_msg_to_homogeneous_tf(chosen_node.transform))
    object_pos_with_ec_orientation[:3, 3] = tra.translation_from_matrix(object_pose)
    pre_grasp_pose_in_base_frame = object_pos_with_ec_orientation.dot(pre_grasp_in_object_frame)

    return pre_grasp_pose_in_base_frame


def get_corner_pregrasp_pose_in_base_frame(chosen_node, pre_grasp_in_object_frame, object_pose, graph_in_base):

    corner_frame = graph_in_base.dot(convert_transform_msg_to_homogeneous_tf(chosen_node.transform))
    ec_frame = get_derived_corner_grasp_frames(corner_frame, object_pose)[0]
    pre_grasp_pose_in_base_frame = ec_frame.dot(pre_grasp_in_object_frame)

    return pre_grasp_pose_in_base_frame


def get_pregrasp_pose_in_base_frame(chosen_node, pre_grasp_in_object_frame, object_pose, graph_in_base):
    if chosen_node.label == 'SurfaceGrasp':
        return get_surface_pregrasp_pose_in_base_frame(pre_grasp_in_object_frame, object_pose)

    elif chosen_node.label == 'WallGrasp':
        return get_wall_pregrasp_pose_in_base_frame(chosen_node, pre_grasp_in_object_frame, object_pose, graph_in_base)

    elif chosen_node.label == 'CornerGrasp':
        return get_corner_pregrasp_pose_in_base_frame(chosen_node, pre_grasp_in_object_frame, object_pose, graph_in_base)

    else:
        raise ValueError("Unknown grasp type: {}".format(chosen_node.label))

