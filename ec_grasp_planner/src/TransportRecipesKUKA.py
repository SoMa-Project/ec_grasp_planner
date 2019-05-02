import numpy as np
import math
import hatools.components as ha
from grasp_success_estimator import RESPONSES

def get_transport_recipe(chosen_object, handarm_params, reaction, FailureCases, grasp_type, handarm_type):
    # Get non grasp-specific specific params    
    joint_damping = handarm_params['joint_damping']
    drop_off_pose = handarm_params['drop_off_pose']
    view_pose = handarm_params['view_pose']
    success_estimator_timeout = handarm_params['success_estimator_timeout']

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params[grasp_type]:
        params = handarm_params[grasp_type][object_type]
    else:
        params = handarm_params[grasp_type]['object']

    lift_time = params['lift_duration']
    up_speed = params['up_speed']
    high_joint_stiffness = params['high_joint_stiffness']
    
    # Up speed is positive because it is defined on the world frame
    up_world_twist = np.array([0, 0, up_speed, 0, 0, 0])

    # Up speed is negative because it is defined on the EE frame
    up_EE_twist = np.array([0, 0, -up_speed, 0, 0, 0])

    # Default srtiffen on Transport
    next_control_mode = 'GoStiff1'

    # assemble controller sequence
    control_sequence = []

    # 1. Lift upwards
    if grasp_type == 'SurfaceGrasp':
        control_sequence.append(ha.CartesianVelocityControlMode(up_EE_twist, controller_name = 'GoUpHTransform', name = 'GoUp_1', reference_frame="EE"))
    elif grasp_type == 'WallGrasp' or grasp_type == 'CornerGrasp':
        control_sequence.append(ha.CartesianVelocityControlMode(up_world_twist, controller_name = 'GoUpHTransform', name = 'GoUp_1', reference_frame="world"))    
    
    # Do not stiffen the arm for the Pisa hand case
    if "pisa_hand" in handarm_type or "clash_hand" in handarm_type:
        next_control_mode = 'PrepareForEstimation'

    # 1b. Switch after the lift time
    control_sequence.append(ha.TimeSwitch('GoUp_1', next_control_mode, duration = lift_time))

    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff1', mode_id = 'joint_impedance', 
                                                        joint_stiffness = high_joint_stiffness, joint_damping = joint_damping))
    # 0b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoStiff1', 'PrepareForEstimation', duration=1.0))

    # 1c. Switch if trik failed
    control_sequence.append(ha.RosTopicSwitch('GoUp_1', next_control_mode, ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 2a. Stay still for a bit
    control_sequence.append(ha.BlockJointControlMode(name='PrepareForEstimation'))

    # 2b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('PrepareForEstimation', 'EstimationMassMeasurement', duration=0.5))

    # 3a. Measure the mass again and estimate number of grasped objects (grasp success estimation)
    control_sequence.append(ha.BlockJointControlMode(name='EstimationMassMeasurement'))

    # 3b. Switches after estimation measurement was done
    target_cm_okay = 'GoDropOff'

    # 3b.1 No object was grasped => go to failure mode.
    target_cm_estimation_no_object = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_NO_OBJECT, default=target_cm_okay)
    if target_cm_estimation_no_object != target_cm_okay:
        target_cm_estimation_no_object = 'softhand_open_after_failure_and_' + target_cm_estimation_no_object
    control_sequence.append(ha.RosTopicSwitch('EstimationMassMeasurement', target_cm_estimation_no_object,
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.ESTIMATION_RESULT_NO_OBJECT.value]),
                                              ))

    # 3b.2 More than one object was grasped => failure
    target_cm_estimation_too_many = reaction.cm_name_for(FailureCases.MASS_ESTIMATION_TOO_MANY, default=target_cm_okay)
    if target_cm_estimation_too_many != target_cm_okay:
        target_cm_estimation_too_many = 'softhand_open_after_failure_and_' + target_cm_estimation_too_many
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

        if "ClashHand" in handarm_type:
            # Open hand goal
            goal_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            control_sequence.append(ha.ros_CLASHhandControlMode(goal=goal_open, behaviour='GotoPos',  name=cm))
        else:
            control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = cm, synergy = 1))

        # Failure control modes representing grasping failure, which might be corrected by re-running the plan or replanning. 
        control_sequence.append(ha.TimeSwitch(cm, 'GoToViewPose_after_failure', duration=0.5))

        # 4. Go to view pose
        control_sequence.append(ha.InterpolatedHTransformControlMode(view_pose, controller_name = 'GoToView', goal_is_relative='0', name = 'GoToViewPose_after_failure', reference_frame = 'world'))
     
        # 4b1. Switch when hand reaches the goal pose
        control_sequence.append(ha.FramePoseSwitch('GoToViewPose_after_failure', 'finished', controller = 'GoToView', epsilon = '0.01'))

        # 4b2. Switch when hand reaches the goal pose
        control_sequence.append(ha.RosTopicSwitch('GoToViewPose_after_failure', 'finished', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([2.])))

        # 4c. Switch if no plan was found
        control_sequence.append(ha.RosTopicSwitch('GoToViewPose_after_failure', 'finished', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))
        
        # 5. Finish the plan
        control_sequence.append(ha.BlockJointControlMode(name='finished'))

    
    # 5. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(drop_off_pose, controller_name = 'GoToDropPose', goal_is_relative='0', name = target_cm_okay, reference_frame = 'world'))
 
    # 5b1. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch(target_cm_okay, 'PlaceInTote', controller = 'GoToDropPose', epsilon = '0.05'))

    # 5b2. Switch when hand reaches the goal pose
    control_sequence.append(ha.RosTopicSwitch(target_cm_okay, 'PlaceInTote', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([2.])))

    # 5c. Switch to recovery if no plan is found
    control_sequence.append(ha.RosTopicSwitch(target_cm_okay, 'recovery_NoPlanFound' + grasp_type, ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    return control_sequence