import numpy as np
# import tf_conversions.posemath as pm
from tf import transformations as tra
from geometry_graph_msgs.msg import geometry_msgs

def fromMsgToMatrix(p):
    """
    :param p: input pose
    :type p: :class:`geometry_msgs.msg.Pose`
    :return: numpy 4x4 array

    Convert a pose represented as a ROS Pose message to a numpy 4x4 array.
    Replaces former operation pm.toMatrix(pm.fromMsg(msg)) (KDL)
    """
    # from quaternion to rotation matrix
    # implementation as in KDL:
    # x2 = p.orientation.x**2
    # y2 = p.orientation.y**2
    # z2 = p.orientation.z**2
    # w2 = p.orientation.w**2
    # return numpy.array([[w2+x2-y2-z2, 2*x*y-2*w*z, 2*x*z+2*w*y, p.position.x],
    #                     [2*x*y+2*w*z, w2-x2+y2-z2, 2*y*z-2*w*x, p.position.y],
    #                     [2*x*z-2*w*y, 2*y*z+2*w*x, w2-x2-y2+z2, p.position.z],
    #                     [0,0,0,1]])
    q = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    return tra.quaternion_matrix(q)

def fromMatrixToMsg(m):
    """
    :param p: input 4x4 matrix
    :type m: :func:`numpy.array`
    :return: ROS Pose message

    Convert a numpy 4x4 array to a pose represented as a ROS Pose message.
    Replaces former operation pm.toMsg(pm.fromMatrix(htf)) (KDL)
    """
    p = Pose()
    # from rotation matrix to quaternion
    q = tra.quaternion_from_matrix(m[0:3][0:3])
    # implementation as in KDL: 
    # trace = m[0,0] + m[1,1] + m[2,2]
    # epsilon=1E-12
    # if trace > epsilon:
    #     s = 0.5 / sqrt(trace + 1.0)
    #     w = 0.25 / s
    #     x = ( m[2,1] - m[1,2] ) * s
    #     y = ( m[0,2] - m[2,0] ) * s
    #     z = ( m[1,0] - m[0,1] ) * s
    # else:
    #     if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
    #         s = 2.0 * sqrt( 1.0 + m[0,0] - m[1,1] - m[2,2])
    #         w = (m[2,1] - m[1,2] ) / s
    #         x = 0.25 * s
    #         y = (m[0,1] + m[1,0] ) / s
    #         z = (m[0,2] + m[2,0] ) / s
    #     elif m[1,1] > m[2,2]:
    #         s = 2.0 * sqrt( 1.0 + m[1,1] - m[0,0] - m[2,2])
    #         w = (m[0,2] - m[2,0] ) / s
    #         x = (m[0,1] + m[1,0] ) / s
    #         y = 0.25 * s
    #         z = (m[1,2] + m[2,1] ) / s
    #     else:
    #         s = 2.0 * sqrt( 1.0 + m[2,2] - m[0,0] - m[1,1] )
    #         w = (m[1,0] - m[0,1] ) / s
    #         x = (m[0,2] + m[2,0] ) / s
    #         y = (m[1,2] + m[2,1] ) / s
    #         z = 0.25 * s
    # p.orientation.x = x
    # p.orientation.y = y 
    # p.orientation.z = z
    # p.orientation.w = w
    
    p.orientation.x = q[0]
    p.orientation.y = q[1]
    p.orientation.z = q[2]
    p.orientation.w = q[3]
    p.position.x = m[0,3]
    p.position.y = m[1,3]
    p.position.z = m[2,3]
    return p

def unit_vector(vector):
    # Returns the unit vector of the vector.
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def convert_pose_msg_to_homogeneous_tf(msg):
    # return pm.toMatrix(pm.fromMsg(msg))
    return fromMsgToMatrix(msg)


def convert_homogeneous_tf_to_pose_msg(htf):
    return fromMatrixToMsg(htf)
    # return pm.toMsg(pm.fromMatrix(htf))


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