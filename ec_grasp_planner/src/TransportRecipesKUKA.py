import numpy as np
import math
import hatools.components as ha
from grasp_success_estimator import RESPONSES


def get_transport_recipe(chosen_object, handarm_params, reaction, FailureCases, grasp_type, alternative_behavior=None):

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params[grasp_type]:
        params = handarm_params[grasp_type][object_type]
    else:
        params = handarm_params[grasp_type]['object']

    lift_time = params['lift_duration']
    up_speed = params['up_speed']
    drop_off_pose = handarm_params['drop_off_pose']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Up speed is positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])

    # assemble controller sequence
    control_sequence = []

    # 1. Lift upwards
    control_sequence.append(ha.CartesianVelocityControlMode(up_twist, controller_name = 'GoUpHTransform', name = 'GoUp_1', reference_frame="world"))
 
    # 1b. Switch after half the lift time
    control_sequence.append(ha.TimeSwitch('GoUp_1', 'EstimationMassMeasurement', duration = lift_time/2))

    # 2. Measure the mass again and estimate number of grasped objects (grasp success estimation)
    control_sequence.append(ha.BlockJointControlMode(name='EstimationMassMeasurement'))

    # 2b. Switches after estimation measurement was done
    target_cm_okay = 'GoUp_2'

    # 2b.1 No object was grasped => go to failure mode.
    target_cm_estimation_no_object = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_NO_OBJECT, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_no_object,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_NO_OBJECT.value]),
                                              ))

    # 2b.2 More than one object was grasped => failure
    target_cm_estimation_too_many = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_TOO_MANY, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_too_many,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_TOO_MANY.value]),
                                              ))

    # 2b.3 Exactly one object was grasped => success (continue lifting the object and go to drop off)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_OKAY.value]),
                                              ))

    # 2b.4 The grasp success estimator module is inactive => directly continue lifting the object and go to drop off
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 2b.5 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('EstimationMassMeasurement', target_cm_okay,
                                          duration=success_estimator_timeout))

    # 2b.6 There is no special switch for unknown error response (estimator signals ESTIMATION_RESULT_UNKNOWN_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.


    # 3. After estimation measurement control modes.
    extra_failure_cms = set()
    if target_cm_estimation_no_object != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_no_object)
    if target_cm_estimation_too_many != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_too_many)

    for cm in extra_failure_cms:
        if cm.startswith('failure_rerun'):
            # 3.1 Failure control mode representing grasping failure, which might be corrected by re-running the plan.
            control_sequence.append(ha.BlockJointControlMode(name=cm))
        if cm.startswith('failure_replan'):
            # 3.2 Failure control mode representing grasping failure, which can't be corrected and requires to re-plan.
            control_sequence.append(ha.BlockJointControlMode(name=cm))

    # 3.3 Success control mode. Lift hand even further
    control_sequence.append(ha.CartesianVelocityControlMode(up_twist, controller_name = 'GoUpHTransform', name = target_cm_okay, reference_frame="world"))
 
    # 3b. Switch after half the lift time
    control_sequence.append(ha.TimeSwitch(target_cm_okay, 'GoDropOff', duration = lift_time/2))
    
    # 4. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(drop_off_pose, controller_name = 'GoToDropPose', goal_is_relative='0', name = 'GoDropOff'))
 
    # 4b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('GoDropOff', 'PlaceInIFCO', controller = 'GoToDropPose', epsilon = '0.01'))

    return control_sequence