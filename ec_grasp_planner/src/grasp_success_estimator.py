#!/usr/bin/env python

import yaml
import rospy
import tf
import tf.transformations
from enum import Enum

# from std_msgs.msg import Int8 <-- Not used anymore since ROSTopicSensor only supports Float64
from std_msgs.msg import Float64
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped
from hybrid_automaton_msgs.msg import HAMState

from scipy.stats import norm

import sys


class RESPONSES(Enum):
    REFERENCE_MEASUREMENT_SUCCESS = 0.0
    REFERENCE_MEASUREMENT_FAILURE = 1.0

    ESTIMATION_RESULT_NO_OBJECT = 10.0
    ESTIMATION_RESULT_OKAY = 11.0
    ESTIMATION_RESULT_TOO_MANY = 12.0
    ESTIMATION_RESULT_UNKNOWN_FAILURE = 13.0

    GRASP_SUCCESS_ESTIMATOR_INACTIVE = 99.0


class ObjectModel(object):
    def __init__(self, name, mass_mean, mass_stddev):
        # The name of the object class (e.g. cucumber)
        self.name = name
        # mean of the object masses (e.g. over all cucumbers) in kg
        self.mass_mean = mass_mean
        # standard deviation of mass of the object (e.g. over all cucumbers)
        self.mass_stddev = mass_stddev


class MassEstimator(object):

    # If confidence is lower than this threshold the estimator warns that confidence is low (will still report a result)
    CONFIDENCE_THRESHOLD = 0.3

    # standard acceleration due to gravity
    GRAVITY = 9.806

    @staticmethod
    def force2mass(f):
        # F = m * a => F/a = m
        return f / -MassEstimator.GRAVITY  # We need the negative value since gravity accelerates along negative z-axis

    @staticmethod
    def load_object_params(path_to_object_parameters):
        reference_object_info = {}
        with open(path_to_object_parameters, 'r') as stream:
            try:
                data = yaml.load(stream)
                for o in data:
                    if 'mass' in data[o]:
                        reference_object_info[o] = ObjectModel(o, data[o]['mass']['mean'], data[o]['mass']['stddev'])
                # print("data loaded {}".format(file))
                return reference_object_info
            except yaml.YAMLError as exc:
                print(exc)
                return None

    @staticmethod
    def list_to_Float64MultiArrayMsg(l):
        l_copy = list(l)
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(size=len(l_copy))]
        msg.data = l_copy

        return msg

    def __init__(self, ft_topic_name, ft_topic_type, object_ros_param_path, path_to_object_parameters):
        rospy.init_node('graspSuccessEstimatorMass', anonymous=True)

        self.tf_listener = tf.TransformListener()

        # Stores the last measurement from the ft sensor. # TODO change this to a sliding window?
        self.last_ft_measurement = None
        # This is the ft measurement that is used as the empty hand reference. From this one we can calculate change.
        self.ft_measurement_reference = None
        # The calculated reference mass in kg.
        self.reference_mass = None
        # Object information loaded on node start-up from the provided object parameter file.
        self.objects_info = MassEstimator.load_object_params(path_to_object_parameters)
        # The name of the ros parameter than contains the name of the current object (e.g. /planner_gui/object).
        self.object_ros_param_path = object_ros_param_path
        # The name of the current object retrieved from ros parameters during reference acquisition and checked
        # during estimation. Used to access object information (e.g. mass probability distribution).
        self.current_object_name = rospy.get_param(self.object_ros_param_path, default=None)

        # Create the publishers
        self.estimator_status_pub = rospy.Publisher('/graspSuccessEstimator/status', Float64, queue_size=10)
        self.estimator_number_pub = rospy.Publisher('/graspSuccessEstimator/num_objects', Float64, queue_size=10)
        self.estimator_confidence_pub = rospy.Publisher('/graspSuccessEstimator/confidence', Float64, queue_size=10)
        self.estimator_confidence_all_pub = rospy.Publisher('/graspSuccessEstimator/confidence_all', Float64MultiArray,
                                                            queue_size=10)
        self.estimator_mass_pub = rospy.Publisher('/graspSuccessEstimator/masses', Float64MultiArray, queue_size=10)

        # Stores the last active control mode to ensure we only start estimation once (the moment we enter the state)
        self.active_cm = None
        self.ham_state_subscriber = rospy.Subscriber("/ham_state", HAMState, self.ham_state_callback)

        # subscriber to the ft topic that is used to estimate the weight
        if ft_topic_type == "Float64MultiArray":
            self.ft_sensor_subscriber = rospy.Subscriber(ft_topic_name, Float64MultiArray,
                                                         self.ft_sensor_float_array_callback)
        elif ft_topic_type == "WrenchStamped":
            self.ft_sensor_subscriber = rospy.Subscriber(ft_topic_name, WrenchStamped, self.ft_sensor_callback)
        else:
            raise ValueError("The given message type {} is not supported.".format(ft_topic_type))

    def ft_sensor_callback(self, data):
        # TODO add a filter that averages a sliding window until all FT measurements in that window have a smaller
        # variance than a predefined threshold value and the sliding window is completely filled up with messages.
        self.last_ft_measurement = data

    def ft_sensor_float_array_callback(self, data):
        last_ft_measurement = WrenchStamped()
        last_ft_measurement.wrench.force.x = data.data[0]
        last_ft_measurement.wrench.force.y = data.data[1]
        last_ft_measurement.wrench.force.z = data.data[2]

        last_ft_measurement.wrench.torque.x = data.data[3]
        last_ft_measurement.wrench.torque.y = data.data[4]
        last_ft_measurement.wrench.torque.z = data.data[5]

        self.last_ft_measurement = last_ft_measurement

    def ham_state_callback(self, data):
        new_cm = data.executing_control_mode_name
        if new_cm != self.active_cm:
            self.active_cm = new_cm

        #    print(new_cm)

            if rospy.get_param("/graspSuccessEstimator/active", default=True):
                if new_cm == 'ReferenceMassMeasurement':
                    self.calculate_reference_mass()
                elif new_cm == 'EstimationMassMeasurement':
                    self.estimate_number_of_objects()
            elif new_cm in ['ReferenceMassMeasurement', 'EstimationMassMeasurement']:
                self.publish_status(RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE)

    def publish_status(self, status):
        rospy.sleep(2.0)  # Otherwise windows side isn't quick enough to subscribe to topic. TODO find better solution?
        self.estimator_status_pub.publish(status.value)

    def to_base_frame(self, ft_measurement_msg):
        try:
            # transform ft sensor frame to world frame which makes it easy to identify the gravity component of
            # the ft sensor values as the z-axis (ft_sensor_wrench is in ee frame)
            (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/ee', rospy.Time(0))

            R = tf.transformations.quaternion_matrix(rot)
            T = tf.transformations.translation_matrix(trans)
            frame_transform = tf.transformations.concatenate_matrices(T, R)

            ft_measurement = [ft_measurement_msg.wrench.force.x, ft_measurement_msg.wrench.force.y,
                              ft_measurement_msg.wrench.force.z, 1]
            # print("4", frame_transform, ft_measurement)
            # print("4", type(frame_transform), type(ft_measurement))
            return frame_transform.dot(ft_measurement)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Could not lookup transform from /ee to /base_link")
            # TODO potentially add wait for transform (with timeout)
            return None

    def calculate_reference_mass(self):

        self.current_object_name = rospy.get_param(self.object_ros_param_path, default=None)

        # print("LAST FT MEASUREMENT", self.last_ft_measurement)
        # TODO maybe add some kind of wait here to make sure the force/torque readings are stable (e.g. variance filter)
        if self.last_ft_measurement is not None:

            ft_in_base = self.to_base_frame(self.last_ft_measurement)
            if ft_in_base is not None:
                self.ft_measurement_reference = ft_in_base
                self.reference_mass = MassEstimator.force2mass(ft_in_base[2])  # z-axis
                print("Reference mass: {}".format(self.reference_mass))
                self.publish_status(RESPONSES.REFERENCE_MEASUREMENT_SUCCESS)
                self.estimator_mass_pub.publish(data=[self.reference_mass])
                return

        # In case we weren't able to calculate the reference mass, signal failure
        self.ft_measurement_reference = None
        rospy.logerr("Reference measurement couldn't be done. Is FT-topic alive?")
        self.publish_status(RESPONSES.REFERENCE_MEASUREMENT_FAILURE)

    def estimate_number_of_objects(self):
        if self.ft_measurement_reference is None:
            rospy.logerr("No reference measurement is present.")
            self.publish_status(RESPONSES.ESTIMATION_RESULT_UNKNOWN_FAILURE)
            return  # failure

        if self.current_object_name is None or self.current_object_name != rospy.get_param(self.object_ros_param_path,
                                                                                           default=None):
            rospy.logerr("The object name of the reference measurement doesn't match the one of the estimation.")
            self.publish_status(RESPONSES.ESTIMATION_RESULT_UNKNOWN_FAILURE)
            return  # failure

        if self.current_object_name not in self.objects_info:

            rospy.logwarn("The current object {0} does not have the required object mass parameter set. Skip it".format(
                self.current_object_name))
            self.publish_status(RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE)
            return  # ignore

        # TODO maybe add some kind of wait here to make sure the force/torque readings are stable (e.g. variance filter)
        # TODO one good way would be to refactor and move code to a ft getter method...
        if self.last_ft_measurement is not None:
            ft_in_base = self.to_base_frame(self.last_ft_measurement)
            if ft_in_base is not None:
                second_mass = MassEstimator.force2mass(ft_in_base[2])  # z-axis
                mass_diff = second_mass - self.reference_mass

                print("MASSES:", second_mass, self.reference_mass, " diff:", mass_diff)
                print("FT_IN_BASE ref:", self.ft_measurement_reference)
                print("FT_IN_BASE est:", ft_in_base)

                object_mean = self.objects_info[self.current_object_name].mass_mean
                object_stddev = self.objects_info[self.current_object_name].mass_stddev

                max_pdf_val = -1.0
                max_num_obj = -1
                pdf_val_sum = 0.0
                pdf_values = []
                for num_obj in range(0, 5):  # classes (number of detected objects) we perform maximum likelihood on.
                    pdf_val = norm.pdf(mass_diff, object_mean * num_obj, object_stddev)
                    pdf_values.append(pdf_val)
                    pdf_val_sum += pdf_val
                    if pdf_val > max_pdf_val:
                        max_pdf_val = pdf_val
                        max_num_obj = num_obj

                confidence = max_pdf_val / pdf_val_sum

                if confidence < MassEstimator.CONFIDENCE_THRESHOLD:
                    rospy.logwarn("Confidence ({0}) is very low! Mass diff was: {1}".format(confidence, mass_diff))

                # publish the number of objects with the highest likelihood and the confidence
                self.estimator_number_pub.publish(max_num_obj)
                self.estimator_confidence_pub.publish(confidence)
                confidence_all = map(lambda x: x / pdf_val_sum, pdf_values)
                self.estimator_confidence_all_pub.publish(MassEstimator.list_to_Float64MultiArrayMsg(confidence_all))

                # publish the estimated masses
                self.estimator_mass_pub.publish(MassEstimator.list_to_Float64MultiArrayMsg(
                    [self.reference_mass, second_mass]))

                # publish the corresponding status message
                if max_num_obj == 0:
                    self.publish_status(RESPONSES.ESTIMATION_RESULT_NO_OBJECT)
                elif max_num_obj == 1:
                    self.publish_status(RESPONSES.ESTIMATION_RESULT_OKAY)
                else:
                    self.publish_status(RESPONSES.ESTIMATION_RESULT_TOO_MANY)
                return  # success

        # In case we weren't able to calculate the number of objects, signal failure
        self.ft_measurement_reference = None
        rospy.logerr("Estimation measurement couldn't be done. Is FT-topic alive?")
        self.publish_status(RESPONSES.ESTIMATION_RESULT_UNKNOWN_FAILURE)


if __name__ == '__main__':

    my_argv = rospy.myargv(argv=sys.argv)
    if len(my_argv) < 5:
        print("usage: grasp_success_estimator.py ft_topic_name ft_topic_type object_ros_param_path path_to_object_parameters")
    else:
        we = MassEstimator(my_argv[1], my_argv[2], my_argv[3], my_argv[4])
        rospy.spin()
        # locking between the callbacks not required as long as we use only one spinner. See:
        # https://answers.ros.org/question/48429/should-i-use-a-lock-on-resources-in-a-listener-node-with-multiple-callbacks/