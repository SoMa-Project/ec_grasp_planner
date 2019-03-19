import rospy
import tf
from std_msgs.msg import Header
import tf.transformations as tra

rospy.init_node("testtest")

tf_listener = tf.TransformListener()

time = rospy.Time(0)
tf_listener.waitForTransform('base_link', "controller_goal_grasp", time, rospy.Duration(2))
A = tf_listener.asMatrix('base_link', Header(0, time, "controller_goal_grasp"))
print(A)

(trans, rot) = tf_listener.lookupTransform("controller_goal_grasp",'base_link', time)
#print(trans, rot)
tf_transformer = tf.TransformerROS()
B = tf_transformer.fromTranslationRotation(trans, rot)
print(B)

C = tra.inverse_matrix(B)
print(C)
