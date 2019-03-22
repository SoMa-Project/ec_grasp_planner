import numpy as np
import tf_conversions.posemath as pm
from tf import transformations as tra


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

