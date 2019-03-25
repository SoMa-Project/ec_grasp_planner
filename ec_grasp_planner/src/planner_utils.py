import numpy as np
import tf_conversions.posemath as pm
from tf import transformations as tra
from geometry_graph_msgs.msg import geometry_msgs


def unit_vector(vector):
    # Returns the unit vector of the vector.
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def convert_pose_msg_to_homogeneous_tf(msg):
    return pm.toMatrix(pm.fromMsg(msg))


def convert_homogeneous_tf_to_pose_msg(htf):
    return pm.toMsg(pm.fromMatrix(htf))


def convert_transform_msg_to_homogeneous_tf(msg):
    return np.dot(tra.translation_matrix([msg.translation.x, msg.translation.y, msg.translation.z]),
                  tra.quaternion_matrix([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))


def tf_dbg_call_to_string(in_transformation, frame_name='dbg'):
    msgframe = transform_to_pose_msg(in_transformation)
    return "rosrun tf static_transform_publisher {0} {1} {2} {3} {4} {5} {6} rlab_origin {7} 100".format(
        msgframe.position.x, msgframe.position.y, msgframe.position.z, msgframe.orientation.x,
        msgframe.orientation.y, msgframe.orientation.z, msgframe.orientation.w, frame_name)


def transform_to_pose(in_transform):
    # convert 4x4 matrix to trans + rot
    scale, shear, angles, translation, persp = tra.decompose_matrix(in_transform)
    orientation_quat = tra.quaternion_from_euler(angles[0], angles[1], angles[2])
    return translation, orientation_quat


def transform_to_pose_msg(in_transform):
    trans, rot = transform_to_pose(in_transform)
    trans = geometry_msgs.msg.Point(x=trans[0], y=trans[1], z=trans[2])
    rot = geometry_msgs.msg.Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
    return geometry_msgs.msg.Pose(position=trans, orientation=rot)