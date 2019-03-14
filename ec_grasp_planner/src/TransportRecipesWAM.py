import numpy as np
import math
import hatools.components as ha
from tf import transformations as tra
from grasp_success_estimator import RESPONSES


def get_transport_recipe(chosen_object, handarm_params, reaction, FailureCases, grasp_type):

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params[grasp_type]:
        params = handarm_params[grasp_type][object_type]
    else:
        params = handarm_params[grasp_type]['object']

    # ############################ #
    # Get params
    # ############################ #

    up_dist = params['up_dist']

    # split the lifted distance into two consecutive lifts (with success estimation in between)
    scale_up = 0.7
    dir_up1 = tra.translation_matrix([0, 0, scale_up * up_dist])
    dir_up2 = tra.translation_matrix([0, 0, (1.0 - scale_up) * up_dist])

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    drop_off_config = handarm_params['drop_off_config']

    # ############################ #
    # Assemble controller sequence
    # ############################ #

    control_sequence = []

    # 1. Lift upwards a little bit (half way up)
    control_sequence.append(ha.InterpolatedHTransformControlMode(dir_up1, controller_name='GoUpHTransform',
                                                                 name='GoUp_1', goal_is_relative='1',
                                                                 reference_frame="world"))

    # 1b. Switch when joint configuration (half way up) is reached
    control_sequence.append(ha.FramePoseSwitch('GoUp_1', 'PrepareForEstimationMassMeasurement',
                                               controller='GoUpHTransform', epsilon='0.01', goal_is_relative='1',
                                               reference_frame="world"))

    # 2. Hold current joint config 
    control_sequence.append(ha.BlockJointControlMode(name='PrepareForEstimationMassMeasurement'))

    # 2b. Wait for a bit to allow vibrations to attenuate # TODO check if this is required...
    control_sequence.append(ha.TimeSwitch('PrepareForEstimationMassMeasurement', 'EstimationMassMeasurement',
                                          duration=0.5))

    # 3. Measure the mass again and estimate number of grasped objects (grasp success estimation)
    control_sequence.append(ha.BlockJointControlMode(name='EstimationMassMeasurement'))

    # 3b. Switches after estimation measurement was done
    target_cm_okay = 'GoUp_2'

    # 3b.1 No object was grasped => go to failure mode.
    target_cm_estimation_no_object = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_NO_OBJECT, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_no_object,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_NO_OBJECT.value]),
                                              ))

    # 3b.2 More than one object was grasped => failure
    target_cm_estimation_too_many = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_TOO_MANY, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_too_many,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_TOO_MANY.value]),
                                              ))

    # 3b.3 Exactly one object was grasped => success (continue lifting the object and go to drop off)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_OKAY.value]),
                                              ))

    # 3b.4 The grasp success estimator module is inactive => directly continue lifting the object and go to drop off
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_okay,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 3b.5 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('EstimationMassMeasurement', target_cm_okay,
                                          duration=success_estimator_timeout))

    # 3b.6 There is no special switch for unknown error response (estimator signals ESTIMATION_RESULT_UNKNOWN_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.

    # 4. After estimation measurement control modes.
    extra_failure_cms = set()
    if target_cm_estimation_no_object != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_no_object)
    if target_cm_estimation_too_many != target_cm_okay:
        extra_failure_cms.add(target_cm_estimation_too_many)

    for cm in extra_failure_cms:
        if cm.startswith('failure_rerun'):
            # 4.1 Failure control mode representing grasping failure, which might be corrected by re-running the plan.
            control_sequence.append(ha.GravityCompensationMode(name=cm))
        if cm.startswith('failure_replan'):
            # 4.2 Failure control mode representing grasping failure, which can't be corrected and requires to re-plan.
            control_sequence.append(ha.GravityCompensationMode(name=cm))

    # 4.3 Success control mode. Lift hand even further
    control_sequence.append(ha.InterpolatedHTransformControlMode(dir_up2, controller_name='GoUpHTransform',
                                                                 name=target_cm_okay, goal_is_relative='1',
                                                                 reference_frame="world"))

    # 4.3b Switch when joint configuration is reached
    control_sequence.append(ha.FramePoseSwitch(target_cm_okay, 'GoDropOff', controller='GoUpHTransform', epsilon='0.01',
                                               goal_is_relative='1', reference_frame="world"))

    # 5. Go to dropOFF location
    control_sequence.append(ha.JointControlMode(drop_off_config, controller_name='GoToDropJointConfig',
                                                name='GoDropOff'))

    # 5.b  Switch when joint is reached
    control_sequence.append(ha.JointConfigurationSwitch('GoDropOff', 'finished', controller = 'GoToDropJointConfig',
                                                        epsilon=str(math.radians(7.))))

    # 6. Block joints to finish motion and hold object in air
    control_sequence.append(ha.BlockJointControlMode(name='finished'))

    return control_sequence