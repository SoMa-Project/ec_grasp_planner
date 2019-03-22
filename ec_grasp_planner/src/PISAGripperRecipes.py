import numpy as np
from tf import transformations as tra
import hatools.components as ha
from grasp_success_estimator import RESPONSES


def create_surface_grasp(chosen_object, handarm_params, pregrasp_transform):
    # Get robot specific params
   

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['SurfaceGrasp']:
        params = handarm_params['SurfaceGrasp'][object_type]
    else:
        params = handarm_params['SurfaceGrasp']['object']
    
    high_cartesian_stiffness = handarm_params['high_cartesian_stiffness']
    low_cartesian_stiffness = params['low_cartesian_stiffness']
    cartesian_damping = handarm_params['cartesian_damping']
    null_stiff = handarm_params['nullspace_stiffness']
    null_damp = handarm_params['nullspace_damping']

    high_joint_stiffness = handarm_params['high_joint_stiffness']
    low_joint_stiffness = params['low_joint_stiffness']
    joint_damping = handarm_params['joint_damping']
    
    # Get params per phase

    # Approach phase
    downward_force = params['downward_force']
    down_speed = params['down_speed']
    hand_preshape_goal = params['hand_preshape_goal']
    hand_preshaping_time = params['hand_preshaping_duration']


    # Grasping phase
    up_speed = params['up_speed']
    hand_closing_time = params['hand_closing_duration']
    hand_closing_goal = params['hand_closing_goal']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the EE frame
    down_twist = np.array([0, 0, down_speed, 0, 0, 0])
    # Slow Up speed is also positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])

    # assemble controller sequence
    control_sequence = []

    # 00a. Change arm mode - soften
    # control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'cartesian_impedance', 
    #                                                     cartesian_stiffness = high_cartesian_stiffness, cartesian_damping = cartesian_damping, 
    #                                                     nullspace_stiffness =null_stiff, nullspace_damping = null_damp))
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'joint_impedance', 
                                                        joint_stiffness = high_joint_stiffness, joint_damping = joint_damping))
    # 00b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoStiff', 'softhand_preshape', duration=1.0))

    # 0. Trigger pre-shaping the hand
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([hand_preshape_goal]), name  = 'softhand_preshape', synergy = '1'))
    
    # 0b. Time to trigger pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape', 'PreGrasp', duration = hand_preshaping_time))

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'PreGrasp', reference_frame = 'world'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForMassMeasurement', controller = 'GoAboveObject', epsilon = '0.03'))
    
    # 1c. Switch to finished if no plan is found
    control_sequence.append(ha.RosTopicSwitch('PreGrasp', 'softhand_open_after_preshape', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 1d. Open hand
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open_after_preshape', synergy = 1))

    # 1e. Wait for a bit and finish
    control_sequence.append(ha.TimeSwitch('softhand_open_after_preshape', 'finished', duration = 0.5))

    # 2. Go to gravity compensation 
    control_sequence.append(ha.BlockJointControlMode(name = 'PrepareForMassMeasurement'))

    # 2b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('PrepareForMassMeasurement', 'ReferenceMassMeasurement', duration = 0.5))

    # 3. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))  # TODO use gravity comp instead?

    # 3b. Switches when reference measurement was done
    # 3b.1 Successful reference measurement
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.REFERENCE_MEASUREMENT_SUCCESS.value]),
                                              ))

    # 3b.2 The grasp success estimator module is inactive
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 3b.3 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('ReferenceMassMeasurement', 'GoDown',
                                          duration=success_estimator_timeout))

    # 3b.4 There is no special switch for unknown error response (estimator signals REFERENCE_MEASUREMENT_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.

    # 4. Go down onto the object (relative in EE frame) - Godown
    control_sequence.append(ha.CartesianVelocityControlMode(down_twist,
                                             controller_name='GoDown',
                                             name="GoDown",
                                             reference_frame="EE"))

    # force threshold that if reached will trigger the closing of the hand
    force = np.array([0, 0, downward_force, 0, 0, 0])
    
    # 4b. Switch when force-torque sensor is triggered
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'GoSoft',
                                                 goal = force,
                                                 norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion = "THRESH_UPPER_BOUND",
                                                 goal_is_relative = '1',
                                                 frame_id = 'world'))

    # 4c. Switch to recovery if the cartesian velocity fails due to joint limits
    control_sequence.append(ha.RosTopicSwitch('GoDown', 'softhand_open_recovery_SurfaceGrasp', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 4d. Change arm mode - soften
    # control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoSoft', mode_id = 'cartesian_impedance', 
    #                                                     cartesian_stiffness = low_cartesian_stiffness, cartesian_damping = cartesian_damping, 
    #                                                     nullspace_stiffness =null_stiff, nullspace_damping = null_damp))
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoSoft', mode_id = 'joint_impedance', 
                                                        joint_stiffness = low_joint_stiffness, joint_damping = joint_damping))
    # 4e. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoSoft', 'softhand_close', duration=1.0))

    # 5. Call hand controller
    if handarm_params['SimplePositionControl']:
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([hand_closing_goal]), name = 'softhand_close', synergy = '1'))
    elif handarm_params['ImpedanceControl']:
        control_sequence.append(ha.ros_PisaIIThandControlMode(goal = np.array([hand_closing_goal]), kp=np.array([params['kp']]), hand_max_aperture = handarm_params['hand_max_aperture'], name  = 'softhand_close', 
            bounding_box=np.array([chosen_object['bounding_box'].x, chosen_object['bounding_box'].y, chosen_object['bounding_box'].z]), object_weight=np.array([0.4]), object_type=object_type, object_pose=chosen_object['frame']))
    else:
        raise Exception("No grasping controller selected for PISA gripper hand")

    # 5b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp_1', duration = hand_closing_time))


    return control_sequence

# ================================================================================================
def create_wall_grasp(chosen_object, wall_frame, handarm_params, pregrasp_transform):
    # Get robot specific params


    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['WallGrasp']:
        params = handarm_params['WallGrasp'][object_type]
    else:
        params = handarm_params['WallGrasp']['object']

    high_cartesian_stiffness = handarm_params['high_cartesian_stiffness']
    cartesian_damping = handarm_params['cartesian_damping']
    low_cartesian_stiffness = params['low_cartesian_stiffness']
    null_stiff = handarm_params['nullspace_stiffness']
    null_damp = handarm_params['nullspace_damping']

    high_joint_stiffness = handarm_params['high_joint_stiffness']
    low_joint_stiffness = params['low_joint_stiffness']
    joint_damping = handarm_params['joint_damping']

    # Get params per phase

    # Approach phase
    downward_force = params['downward_force']
    down_speed = params['down_speed']
    hand_preshape_goal = params['hand_preshape_goal']
    hand_preshaping_time = params['hand_preshaping_duration']

    lift_time = params['corrective_lift_duration']
    up_speed = params['up_speed']

    wall_force = params['wall_force']
    slide_speed = params['slide_speed']

    # Grasping phase
    pre_grasp_twist = params['pre_grasp_twist']
    pre_grasp_rotate_time = params['pre_grasp_rotation_duration']
    hand_closing_time = params['hand_closing_duration']
    hand_closing_goal = params['hand_closing_goal']

    # Post-grasping phase
    post_grasp_twist = params['post_grasp_twist']
    post_grasp_rotate_time = params['post_grasp_rotation_duration']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the twists to use TRIK controller with

    # Down speed is negative because it is defined on the world frame
    down_twist = np.array([0, 0, -down_speed, 0, 0, 0])
    # Slow Up speed is positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])
    # Calculate the twist to slide towards the wall in the world frame
    # This is done because EE frame sliding is no longer safe because of reachability issues
    slide_forwards_linear_velocity = wall_frame[:3,:3].dot(np.array([0, 0, -slide_speed]))
    slide_twist = np.array([slide_forwards_linear_velocity[0], slide_forwards_linear_velocity[1], slide_forwards_linear_velocity[2], 0, 0, 0])

    control_sequence = []

    # 00a. Change arm mode - soften
    # control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'cartesian_impedance', 
    #                                                     cartesian_stiffness = high_cartesian_stiffness, cartesian_damping = cartesian_damping, 
    #                                                     nullspace_stiffness =null_stiff, nullspace_damping = null_damp))
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'joint_impedance', 
                                                        joint_stiffness = high_joint_stiffness, joint_damping = joint_damping))
    # 00b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoStiff', 'softhand_preshape', duration=1.0))

    # 0 trigger pre-shaping the hand
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([hand_preshape_goal]), name  = 'softhand_preshape', synergy = '1'))
    
    # 0b. Time to trigger pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape', 'PreGrasp', duration = hand_preshaping_time))

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'PreGrasp', reference_frame = 'world'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForMassMeasurement', controller = 'GoAboveObject', epsilon = '0.03'))
    
    # 1c. Switch to finished if no plan is found
    control_sequence.append(ha.RosTopicSwitch('PreGrasp', 'softhand_open_after_preshape', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 1d. Open hand
    control_sequence.append(ha.BlockJointControlMode(name  = 'softhand_open_after_preshape'))

    # 1e. Wait for a bit and finish
    control_sequence.append(ha.TimeSwitch('softhand_open_after_preshape', 'finished', duration = 0.5))

    # 2. Go to gravity compensation 
    control_sequence.append(ha.BlockJointControlMode(name = 'PrepareForMassMeasurement'))

    # 2b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('PrepareForMassMeasurement', 'ReferenceMassMeasurement', duration = 0.5))

    # 3. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))  # TODO use gravity comp instead?

    # 3b. Switches when reference measurement was done
    # 3b.1 Successful reference measurement
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.REFERENCE_MEASUREMENT_SUCCESS.value]),
                                              ))

    # 3b.2 The grasp success estimator module is inactive
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 3b.3 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('ReferenceMassMeasurement', 'GoDown',
                                          duration=success_estimator_timeout))

    # 3b.4 There is no special switch for unknown error response (estimator signals REFERENCE_MEASUREMENT_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.


    # 4. Go down onto the object/table, in world frame
    control_sequence.append( ha.CartesianVelocityControlMode(down_twist,
                                             controller_name='GoDown',
                                             name="GoDown",
                                             reference_frame="world"))


    # 4b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'LiftHand',
                                                 goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world'))

    # 4c. Switch to recovery if the cartesian velocity fails due to joint limits
    control_sequence.append(ha.RosTopicSwitch('GoDown', 'softhand_open_recovery_WallGrasp', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 5. Lift upwards so the hand doesn't slide on table surface
    control_sequence.append(
        ha.CartesianVelocityControlMode(up_twist, controller_name='Lift1', name="LiftHand",
                                             reference_frame="world"))

    # 5b. We switch after a short time as this allows us to do a small, precise lift motion
    control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=lift_time))

    # 6. Go towards the wall to slide object to wall
    control_sequence.append(
        ha.CartesianVelocityControlMode(slide_twist, controller_name='SlideToWall',
                                             name="SlideToWall", reference_frame="world"))

    # 6b. Switch when the f/t sensor is triggered with normal force from wall
    force = np.array([0, 0, wall_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', 'GoSoft', 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame))

    # 6c. Switch to recovery if the cartesian velocity fails due to joint limits
    control_sequence.append(ha.RosTopicSwitch('SlideToWall', 'recovery_SlideWG', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 6d. Change arm mode - soften
    # control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoSoft', mode_id = 'cartesian_impedance', 
    #                                                     cartesian_stiffness = low_cartesian_stiffness, cartesian_damping = cartesian_damping, 
    #                                                     nullspace_stiffness =null_stiff, nullspace_damping = null_damp))
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoSoft', mode_id = 'joint_impedance', 
                                                        joint_stiffness = low_joint_stiffness, joint_damping = joint_damping))
    #  6e. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoSoft', 'SlideBackFromWall', duration=1.0))

    # 7. Go back a bit to allow the hand to inflate
    control_sequence.append(
        ha.CartesianVelocityControlMode(pre_grasp_twist, controller_name='SlideBackFromWall',
                                             name="SlideBackFromWall", reference_frame="EE"))
    # 7b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('SlideBackFromWall', 'softhand_close', duration=pre_grasp_rotate_time))
    
    # 8. Call general hand controller
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([hand_closing_goal]), name  = 'softhand_close', synergy = '1'))

    # 8b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration=hand_closing_time))

    # 9. Rotate a bit to roll the object in the hand
    control_sequence.append(
        ha.CartesianVelocityControlMode(post_grasp_twist, controller_name='PostGraspRotate',
                                             name="PostGraspRotate", reference_frame="EE"))
    # 9b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('PostGraspRotate', 'GoUp_1', duration=post_grasp_rotate_time))
    
    return control_sequence

    # ================================================================================================
def create_corner_grasp(chosen_object, corner_frame_alpha_zero, handarm_params, pregrasp_transform):
    # Get robot specific params

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['CornerGrasp']:
        params = handarm_params['CornerGrasp'][object_type]
    else:
        params = handarm_params['CornerGrasp']['object']

    high_cartesian_stiffness = handarm_params['high_cartesian_stiffness']
    cartesian_damping = handarm_params['cartesian_damping']
    low_cartesian_stiffness = params['low_cartesian_stiffness']
    null_stiff = handarm_params['nullspace_stiffness']
    null_damp = handarm_params['nullspace_damping']
    
    high_joint_stiffness = handarm_params['high_joint_stiffness']
    low_joint_stiffness = params['low_joint_stiffness']
    joint_damping = handarm_params['joint_damping']

    # Get params per phase

    # Approach phase
    downward_force = params['downward_force']
    down_speed = params['down_speed']
    hand_preshape_goal = params['hand_preshape_goal']
    hand_preshaping_time = params['hand_preshaping_duration']

    lift_time = params['corrective_lift_duration']
    up_speed = params['up_speed']

    wall_force = params['wall_force']
    slide_speed = params['slide_speed']

    # Grasping phase
    pre_grasp_twist = params['pre_grasp_twist']
    pre_grasp_rotate_time = params['pre_grasp_rotation_duration']
    hand_closing_time = params['hand_closing_duration']
    hand_closing_goal = params['hand_closing_goal']

    # Post-grasping phase
    post_grasp_twist = params['post_grasp_twist']
    post_grasp_rotate_time = params['post_grasp_rotation_duration']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the twists to use TRIK controller with

    # Down speed is negative because it is defined on the world frame
    down_twist = np.array([0, 0, -down_speed, 0, 0, 0])
    # Slow Up speed is positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])
    # Calculate the twist to slide towards the wall in the world frame
    # This is done because EE frame sliding is no longer safe because of reachability issues
    slide_forwards_linear_velocity = corner_frame_alpha_zero[:3,:3].dot(np.array([0, 0, -slide_speed]))
    slide_twist = np.array([slide_forwards_linear_velocity[0], slide_forwards_linear_velocity[1], slide_forwards_linear_velocity[2], 0, 0, 0])

    control_sequence = []
    # 00a. Change arm mode - soften
    # control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'cartesian_impedance', 
    #                                                     cartesian_stiffness = high_cartesian_stiffness, cartesian_damping = cartesian_damping, 
    #                                                     nullspace_stiffness =null_stiff, nullspace_damping = null_damp))
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'joint_impedance', 
                                                        joint_stiffness = high_joint_stiffness, joint_damping = joint_damping))
    # 00b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoStiff', 'softhand_preshape', duration=1.0))

    # 0 trigger pre-shaping the hand
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([hand_preshape_goal]), name  = 'softhand_preshape', synergy = '1'))
    
    # 0b. Time to trigger pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape', 'PreGrasp', duration = hand_preshaping_time))

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'PreGrasp', reference_frame = 'world'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForMassMeasurement', controller = 'GoAboveObject', epsilon = '0.03'))
    
    # 1c. Switch to finished if no plan is found
    control_sequence.append(ha.RosTopicSwitch('PreGrasp', 'softhand_open_after_preshape', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 1d. Open hand
    control_sequence.append(ha.BlockJointControlMode(name  = 'softhand_open_after_preshape'))

    # 1e. Wait for a bit and finish
    control_sequence.append(ha.TimeSwitch('softhand_open_after_preshape', 'finished', duration = 0.5))

    # 2. Go to gravity compensation 
    control_sequence.append(ha.BlockJointControlMode(name = 'PrepareForMassMeasurement'))

    # 2b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('PrepareForMassMeasurement', 'ReferenceMassMeasurement', duration = 0.5))

    # 3. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))  # TODO use gravity comp instead?

    # 3b. Switches when reference measurement was done
    # 3b.1 Successful reference measurement
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.REFERENCE_MEASUREMENT_SUCCESS.value]),
                                              ))

    # 3b.2 The grasp success estimator module is inactive
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 3b.3 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('ReferenceMassMeasurement', 'GoDown',
                                          duration=success_estimator_timeout))

    # 3b.4 There is no special switch for unknown error response (estimator signals REFERENCE_MEASUREMENT_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.


    # 4. Go down onto the object/table, in world frame
    control_sequence.append( ha.CartesianVelocityControlMode(down_twist,
                                             controller_name='GoDown',
                                             name="GoDown",
                                             reference_frame="world"))


    # 4b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'LiftHand',
                                                 goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world'))

    # 4c. Switch to recovery if the cartesian velocity fails due to joint limits
    control_sequence.append(ha.RosTopicSwitch('GoDown', 'softhand_open_recovery_WallGrasp', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 5. Lift upwards so the hand doesn't slide on table surface
    control_sequence.append(
        ha.CartesianVelocityControlMode(up_twist, controller_name='Lift1', name="LiftHand",
                                             reference_frame="world"))

    # 5b. We switch after a short time as this allows us to do a small, precise lift motion
    control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=lift_time))

    # 6. Go towards the wall to slide object to wall
    control_sequence.append(
        ha.CartesianVelocityControlMode(slide_twist, controller_name='SlideToWall',
                                             name="SlideToWall", reference_frame="world"))

    # 6b. Switch when the f/t sensor is triggered with normal force from wall
    force = np.array([0, 0, wall_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', 'GoSoft', 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=corner_frame_alpha_zero))

    # 6c. Switch to recovery if the cartesian velocity fails due to joint limits
    control_sequence.append(ha.RosTopicSwitch('SlideToWall', 'recovery_SlideWG', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))
    
    # 6d. Change arm mode - soften
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoSoft', mode_id = 'cartesian_impedance', 
                                                        cartesian_stiffness = low_cartesian_stiffness, cartesian_damping = cartesian_damping, 
                                                        nullspace_stiffness =null_stiff, nullspace_damping = null_damp))
    # 6e. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoSoft', 'SlideBackFromWall', duration=1.0))

    # 7. Go back a bit to allow the hand to inflate
    control_sequence.append(
        ha.CartesianVelocityControlMode(pre_grasp_twist, controller_name='SlideBackFromWall',
                                             name="SlideBackFromWall", reference_frame="EE"))
    # 7b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('SlideBackFromWall', 'softhand_close', duration=pre_grasp_rotate_time))
    
    # 8. Call general hand controller
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([hand_closing_goal]), name  = 'softhand_close', synergy = '1'))

    # 8b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration=hand_closing_time))

    # 9. Rotate a bit to roll the object in the hand
    control_sequence.append(
        ha.CartesianVelocityControlMode(post_grasp_twist, controller_name='PostGraspRotate',
                                             name="PostGraspRotate", reference_frame="EE"))
    # 9b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('PostGraspRotate', 'GoUp_1', duration=post_grasp_rotate_time))
    
    return control_sequence