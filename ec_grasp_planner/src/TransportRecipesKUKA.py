import numpy as np
import math
import hatools.components as ha
from grasp_success_estimator import RESPONSES

def get_transport_recipe(chosen_object, handarm_params, reaction, FailureCases, grasp_type, handarm_type):

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params[grasp_type]:
        params = handarm_params[grasp_type][object_type]
    else:
        params = handarm_params[grasp_type]['object']

    lift_time = params['lift_duration']
    up_speed = params['up_speed']
    drop_off_pose = handarm_params['drop_off_pose']
    view_joint_config = handarm_params['view_joint_config']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Up speed is positive because it is defined on the world frame
    up_world_twist = np.array([0, 0, up_speed, 0, 0, 0])

    # Up speed is negative because it is defined on the EE frame
    up_EE_twist = np.array([0, 0, -up_speed, 0, 0, 0])

    # assemble controller sequence
    control_sequence = []

    # 1. Lift upwards
    if grasp_type == 'SurfaceGrasp':
        control_sequence.append(ha.CartesianVelocityControlMode(up_EE_twist, controller_name = 'GoUpHTransform', name = 'GoUp_1', reference_frame="EE"))
    elif grasp_type == 'WallGrasp':
        control_sequence.append(ha.CartesianVelocityControlMode(up_world_twist, controller_name = 'GoUpHTransform', name = 'GoUp_1', reference_frame="world"))
    
    # 1b. Switch after the lift time
    control_sequence.append(ha.TimeSwitch('GoUp_1', 'EstimationMassMeasurement', duration = lift_time))

    # 2. Measure the mass again and estimate number of grasped objects (grasp success estimation)
    control_sequence.append(ha.BlockJointControlMode(name='EstimationMassMeasurement'))

    # 2b. Switches after estimation measurement was done
    target_cm_okay = 'GoDropOff'

    # 2b.1 No object was grasped => go to failure mode.
    target_cm_estimation_no_object = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_NO_OBJECT, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', 'softhand_open_after_failure',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_NO_OBJECT.value]),
                                              ))

    # 2b.2 More than one object was grasped => failure
    target_cm_estimation_too_many = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_TOO_MANY, default=target_cm_okay)
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', 'softhand_open_after_failure',
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
        if cm.startswith('failure_rerun') or cm.startswith('failure_replan'):
            # Failure control modes representing grasping failure, which might be corrected by re-running the plan or replanning. 
            control_sequence.append(ha.TimeSwitch('softhand_open_after_failure', 'go_to_view_config_after_failure', duration=0.5))

            # 4. View config above ifco
            control_sequence.append(ha.PlanningJointControlMode(view_joint_config, name='go_to_view_config_after_failure', controller_name='viewJointCtrl'))

            # 4b. Joint config switch
            control_sequence.append(ha.JointConfigurationSwitch('go_to_view_config_after_failure', cm, controller='viewJointCtrl', epsilon=str(math.radians(7.))))

            # 4c. Switch if no plan was found
            control_sequence.append(ha.RosTopicSwitch('go_to_view_config_after_failure', cm, ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))
            
            # 5. Finish the plan
            control_sequence.append(ha.BlockJointControlMode(name=cm))

    if "ClashHand" in handarm_type:
        # Load the proper params from handarm_parameters.py
        # Replace the BlockingJointControlMode with the CLASH hand control mode
        # Open hand goal
        goal_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        control_sequence.append(ha.ros_CLASHhandControlMode(goal=goal_open, behaviour='GotoPos',  name='softhand_open_after_failure'))
    else:
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open_after_failure', synergy = 1))
        
    
    # 4. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(drop_off_pose, controller_name = 'GoToDropPose', goal_is_relative='0', name = target_cm_okay))
 
    # 4b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch(target_cm_okay, 'PlaceInTote', controller = 'GoToDropPose', epsilon = '0.01'))

    # 4c. Switch to recovery if no plan is found
    control_sequence.append(ha.RosTopicSwitch(target_cm_okay, 'recovery_NoPlanFound' + grasp_type, ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))


    return control_sequence