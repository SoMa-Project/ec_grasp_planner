#!/usr/bin/env python

import yaml
import rospy
import tf
import tf.transformations
from enum import Enum
from collections import deque
from functools import partial

import handarm_parameters

# from std_msgs.msg import Int8 <-- Not used anymore since ROSTopicSensor only supports Float64
from std_msgs.msg import Float64
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Wrench, Vector3
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


class EstimationModes(Enum):
    # Use a reference measurement (in a similar pose) and compare against that (more stable)
    DIFFERENCE = 0
    # Just one absolute measurement. Might be more prone to robot noise/ calibration errors.
    ABSOLUTE = 1


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

    def load_robot_noise_params(self):
            hand = rospy.get_param('/planner_gui/hand', default='')
            robot = rospy.get_param('/planner_gui/robot', default='')

            if hand and robot:
                handarm_type = hand + robot
                handarm_params = handarm_parameters.__dict__[handarm_type]()

                mean = handarm_params['success_estimation_robot_noise'][0]
                stddev = handarm_params['success_estimation_robot_noise'][1]

                self.robot_noise = ObjectModel("robot_noise", mean, stddev)
                return True  # success

            rospy.logwarn("Could not load robot noise parameters")
            return False  # failure: params not loaded

    def get_robot_noise_params(self):
        if self.robot_noise is None:
            # no params loaded yet
            if self.load_robot_noise_params():
                return self.robot_noise

            # no params loaded yet. return default ones
            return ObjectModel("default_robot_noise", 0, 0.15)

        # return stored robot noise
        return self.robot_noise

    def __init__(self, ft_topic_name, ft_topic_type, object_ros_param_path, path_to_object_parameters, ee_frame, model):
        rospy.init_node('graspSuccessEstimatorMass', anonymous=True)

        self.tf_listener = tf.TransformListener()

        # Stores the last measurements from the ft sensor (sliding window)
        self.window_size = 25 # TODO come up with somthing reasonable (depending on frequency of ft-sensor... our sends @ 50Hz)
        self.avg_window_ft_measurements = deque(
            [Wrench(force=Vector3(0, 0, 0), torque=Vector3(0, 0, 0)) for i in range(0, self.window_size)])
        self.current_wrench_sum = Wrench(force=Vector3(0, 0, 0), torque=Vector3(0, 0, 0))
        self.current_received_msgs = 0

        self.robot_noise = None

        # Check on what model the estimation should be based on
        try:
            self.model = EstimationModes[model.upper()]
        except KeyError:
            raise ValueError("IllegalArgument: MassEstimator model {} not supported".format(model))

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

        self.ee_frame = ee_frame

        # Create the publishers
        self.estimator_status_pub = rospy.Publisher('/graspSuccessEstimator/status', Float64, queue_size=10)
        self.estimator_number_pub = rospy.Publisher('/graspSuccessEstimator/num_objects', Float64, queue_size=10)
        self.estimator_confidence_pub = rospy.Publisher('/graspSuccessEstimator/confidence', Float64, queue_size=10)
        self.estimator_confidence_all_pub = rospy.Publisher('/graspSuccessEstimator/confidence_all', Float64MultiArray,
                                                            queue_size=10)
        self.estimator_mass_pub = rospy.Publisher('/graspSuccessEstimator/masses', Float64MultiArray, queue_size=10)

        self.estimator_continues_mass_pub = rospy.Publisher('/graspSuccessEstimator/continues_mass', Float64, queue_size=10)

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
            raise ValueError("MassEstimator: The given message type {} is not supported.".format(ft_topic_type))

    def add_latest_ft_measurement(self, last_ft_measurement):

        latest_ft_in_base = self.to_base_frame(last_ft_measurement)

        if latest_ft_in_base is None:
            rospy.logwarn("MassEstimator: Skipped adding ft information. Could not transform to base")
            return

        oldest_ft_measure = self.avg_window_ft_measurements.popleft()

        # TODO refactor: create wrench class with add function or similar.
        self.current_wrench_sum.force.x += latest_ft_in_base[0] - oldest_ft_measure.force.x
        self.current_wrench_sum.force.y += latest_ft_in_base[1] - oldest_ft_measure.force.y
        self.current_wrench_sum.force.z += latest_ft_in_base[2] - oldest_ft_measure.force.z

        #self.current_wrench_sum.torque.x += latest_ft_in_base.wrench.torque.x - oldest_ft_measure.torque.x
        #self.current_wrench_sum.torque.y += latest_ft_in_base.wrench.torque.y - oldest_ft_measure.torque.y
        #self.current_wrench_sum.torque.z += latest_ft_in_base.wrench.torque.z - oldest_ft_measure.torque.z

        latest_wrench = Wrench(force=Vector3(latest_ft_in_base[0], latest_ft_in_base[1], latest_ft_in_base[2]),
                               torque=Vector3(0, 0, 0))  # TODO remove torque, or qctually compute it...

        self.avg_window_ft_measurements.append(latest_wrench)

        self.current_received_msgs = min(self.current_received_msgs + 1, self.window_size)

        self.estimator_continues_mass_pub.publish(MassEstimator.force2mass(self.get_current_ft_estimation().force.z))

    # calculates the current ft estimation based on the avg window etc.
    def get_current_ft_estimation(self):

        if self.current_received_msgs > 0:
            avg_force = Vector3(self.current_wrench_sum.force.x / float(self.current_received_msgs),
                                self.current_wrench_sum.force.y / float(self.current_received_msgs),
                                self.current_wrench_sum.force.z / float(self.current_received_msgs))

            # TODO remove avg_torque since not used anyway right now...
            avg_torque = Vector3(self.current_wrench_sum.torque.x / float(self.current_received_msgs),
                                 self.current_wrench_sum.torque.y / float(self.current_received_msgs),
                                 self.current_wrench_sum.torque.z / float(self.current_received_msgs))

            return Wrench(force=avg_force, torque=avg_torque)

        else:
            return None

    def ft_sensor_callback(self, data):
        self.add_latest_ft_measurement(data)

    def ft_sensor_float_array_callback(self, data):
        last_ft_measurement = WrenchStamped()
        last_ft_measurement.wrench.force.x = data.data[0]
        last_ft_measurement.wrench.force.y = data.data[1]
        last_ft_measurement.wrench.force.z = data.data[2]

        last_ft_measurement.wrench.torque.x = data.data[3]
        last_ft_measurement.wrench.torque.y = data.data[4]
        last_ft_measurement.wrench.torque.z = data.data[5]

        self.add_latest_ft_measurement(last_ft_measurement)

    def ham_state_callback(self, data):
        new_cm = data.executing_control_mode_name
        if new_cm != self.active_cm:
            self.active_cm = new_cm

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
            (trans, rot) = self.tf_listener.lookupTransform('/base_link', self.ee_frame, rospy.Time(0))

            # This is a reminder that ignoring the translational component is intentional...
            # The force transformation should not be affected by the translational component of
            # the base_link -> ee_frame tf (see: https://github.com/SoMa-Project/ec_grasp_planner/issues/56)
            # So the below code
            #   R = tf.transformations.quaternion_matrix(rot)
            #   T = tf.transformations.translation_matrix(trans)
            #   frame_transform = tf.transformations.concatenate_matrices(T, R)
            # can be simplified to:
            frame_transform = tf.transformations.quaternion_matrix(rot)

            ft_measurement = [ft_measurement_msg.wrench.force.x, ft_measurement_msg.wrench.force.y,
                              ft_measurement_msg.wrench.force.z, 1]

            return frame_transform.dot(ft_measurement)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("MassEstimator: Could not lookup transform from /ee to /base_link")
            # TODO potentially add wait for transform (with timeout)
            return None

    def calculate_reference_mass(self):

        self.current_object_name = rospy.get_param(self.object_ros_param_path, default=None)

        if self.current_received_msgs > 0:
            self.ft_measurement_reference = self.get_current_ft_estimation().force
            self.reference_mass = MassEstimator.force2mass(self.ft_measurement_reference.z)  # z-axis
            print("Reference mass: {}".format(self.reference_mass))
            self.publish_status(RESPONSES.REFERENCE_MEASUREMENT_SUCCESS)
            self.estimator_mass_pub.publish(data=[self.reference_mass])
            return

        # In case we weren't able to calculate the reference mass, signal failure
        self.ft_measurement_reference = None
        rospy.logerr("MassEstimator: Reference measurement couldn't be done. Is FT-topic alive?")
        self.publish_status(RESPONSES.REFERENCE_MEASUREMENT_FAILURE)

    def estimate_number_of_objects(self):

        if self.model == EstimationModes.DIFFERENCE:

            if self.ft_measurement_reference is None:
                rospy.logerr("MassEstimator: No reference measurement is present.")
                self.publish_status(RESPONSES.ESTIMATION_RESULT_UNKNOWN_FAILURE)
                return  # failure

            if self.current_object_name is None or self.current_object_name != rospy.get_param(self.object_ros_param_path,
                                                                                               default=None):
                rospy.logerr("MassEstimator: The object name of the reference measurement doesn't match the one of the "
                             "estimation.")
                self.publish_status(RESPONSES.ESTIMATION_RESULT_UNKNOWN_FAILURE)
                return  # failure

        if self.current_object_name not in self.objects_info:

            rospy.logwarn("MassEstimator: The current object {0} does not have the required object mass parameter set."
                          " Skip it".format( self.current_object_name))
            self.publish_status(RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE)
            return  # ignore

        # TODO check even more if force/torque readings are stable (e.g. variance filter)?
        current_ft_estimate = self.get_current_ft_estimation()

        if current_ft_estimate is not None:
            current_mass = MassEstimator.force2mass(current_ft_estimate.force.z)  # z-axis
            reference_mass = self.reference_mass if self.model == EstimationModes.DIFFERENCE else 0.0
            mass_diff = current_mass - reference_mass

            print("MASSES:", current_mass, reference_mass, " diff:", mass_diff, " model:", str(self.model))
            print("FT_IN_BASE ref:", self.ft_measurement_reference)
            print("FT_IN_BASE est:", current_ft_estimate)

            current_object = self.objects_info[self.current_object_name]

            # robot specific parameters (used to check for no object)
            robot_noise = self.get_robot_noise_params()

            if self.model == EstimationModes.DIFFERENCE:
                pdf_fun = partial(MassEstimator.pdf_difference_model, reference_mass)
            else:
                pdf_fun = MassEstimator.pdf_absolute_model

            # check for no object first (robot specific parameters)
            max_pdf_val = pdf_fun(0, current_mass, current_object, robot_noise)
            max_num_obj = 0
            pdf_val_sum = max_pdf_val
            pdf_values = [max_pdf_val]

            # check for number of objects > 0
            # basic object distribution (mean gets shifted by number of objects that are checked against)
            for num_obj in range(1, 5):  # classes (number of detected objects) we perform maximum likelihood on.
                pdf_val = pdf_fun(num_obj, current_mass, current_object, robot_noise)
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
            self.estimator_mass_pub.publish(MassEstimator.list_to_Float64MultiArrayMsg([reference_mass, current_mass]))

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
        rospy.logerr("MassEstimator: Estimation measurement couldn't be done. Is FT-topic alive?")
        self.publish_status(RESPONSES.ESTIMATION_RESULT_UNKNOWN_FAILURE)

    # Probability density function for the difference model (Compare the mass difference between a reference measurement
    # in a similar hand pose with a second one)
    #
    # IMPORTANT: To calculate the correct parameters set the create_absolute_distributions parameter in the calculation
    #            script (calculate_success_estimator_object_params.py, in soma_utils) to False.
    @staticmethod
    def pdf_difference_model(reference_mass, number_of_objects, current_mass, current_object, robot_noise):

        mass_diff = current_mass - reference_mass

        if number_of_objects == 0:
            return norm.pdf(mass_diff, 0, robot_noise.mass_stddev)

        return norm.pdf(mass_diff, current_object.mass_mean * number_of_objects, current_object.mass_stddev)

    # Probability density function for the absolute model (No reference measurement is taken, instead the current
    # mass measurement is directly used).
    #
    # IMPORTANT: This function assumes that the robot noise is already incorporated into the mass distribution of the
    #            object. To do so you can set the create_absolute_distributions parameter in the calculation script
    #            (calculate_success_estimator_object_params.py, in soma_utils) to True.
    @staticmethod
    def pdf_absolute_model(number_of_objects, current_mass, current_object, robot_noise):

        if number_of_objects == 0:
            return norm.pdf(current_mass, robot_noise.mass_mean, robot_noise.mass_stddev)

        mass_mean = current_object.mass_mean * number_of_objects - robot_noise.mass_mean * (number_of_objects-1)
        return norm.pdf(current_mass, mass_mean, current_object.mass_stddev)


if __name__ == '__main__':

    my_argv = rospy.myargv(argv=sys.argv)
    if len(my_argv) < 7:
        print("usage: grasp_success_estimator.py ft_topic_name ft_topic_type object_ros_param_path path_to_object_parameters ee_frame mass_model")
    else:
        we = MassEstimator(my_argv[1], my_argv[2], my_argv[3], my_argv[4], my_argv[5], my_argv[6])
        rospy.spin()
        # locking between the callbacks not required as long as we use only one spinner. See:
        # https://answers.ros.org/question/48429/should-i-use-a-lock-on-resources-in-a-listener-node-with-multiple-callbacks/
