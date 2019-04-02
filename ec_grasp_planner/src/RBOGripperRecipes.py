import numpy as np
from tf import transformations as tra
import hatools.components as ha
from grasp_success_estimator import RESPONSES


def create_surface_grasp(chosen_object, handarm_params, pregrasp_transform):
    # Get robot specific params
    joint_damping = handarm_params['joint_damping']
    success_estimator_timeout = handarm_params['success_estimator_timeout']

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['SurfaceGrasp']:
        params = handarm_params['SurfaceGrasp'][object_type]
    else:
        params = handarm_params['SurfaceGrasp']['object']

    high_joint_stiffness = params['high_joint_stiffness']
    low_joint_stiffness = params['low_joint_stiffness']
    
    # Get params per phase

    # Approach phase
    downward_force = params['downward_force']
    down_speed = params['down_speed']

    # Grasping phase
    up_speed = params['up_speed']
    hand_closing_time = params['hand_closing_duration']

    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the EE frame
    down_twist = np.array([0, 0, down_speed, 0, 0, 0])
    # Slow Up speed is also positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])

    # assemble controller sequence
    control_sequence = []

    # 0a. Change arm mode - soften
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'joint_impedance', 
                                                        joint_stiffness = high_joint_stiffness, joint_damping = joint_damping))
    # 0b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoStiff', 'PreGrasp', duration=1.0))

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'PreGrasp', reference_frame = 'world'))
 
    # 1b1. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'softhand_preshape_1_1', controller = 'GoAboveObject', epsilon = '0.01'))
    
    # 1b2. Switch when hand reaches the goal pose
    control_sequence.append(ha.RosTopicSwitch('PreGrasp', 'softhand_preshape_1_1', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([2.])))

    # 1c. Switch to finished if no plan is found
    control_sequence.append(ha.RosTopicSwitch('PreGrasp', 'finished', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 2a. trigger pre-shaping the hand
    control_sequence.append(ha.BlockJointControlMode(name = 'softhand_preshape_1_1'))

    # 2b. Time to trigger pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_1_1', 'PrepareForMassMeasurement', duration = hand_closing_time))

    # 3. Go to gravity compensation 
    control_sequence.append(ha.BlockJointControlMode(name = 'PrepareForMassMeasurement'))

    # 3b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('PrepareForMassMeasurement', 'ReferenceMassMeasurement', duration = 0.5))

    # 4. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))  # TODO use gravity comp instead?

    # 4b. Switches when reference measurement was done
    # 4b.1 Successful reference measurement
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.REFERENCE_MEASUREMENT_SUCCESS.value]),
                                              ))

    # 4b.2 The grasp success estimator module is inactive
    control_sequence.append(ha.RosTopicSwitch('ReferenceMassMeasurement', 'GoDown',
                                              ros_topic_name='/graspSuccessEstimator/status', ros_topic_type='Float64',
                                              goal=np.array([RESPONSES.GRASP_SUCCESS_ESTIMATOR_INACTIVE.value]),
                                              ))

    # 4b.3 Timeout (grasp success estimator module not started, an error occurred or it takes too long)
    control_sequence.append(ha.TimeSwitch('ReferenceMassMeasurement', 'GoDown',
                                          duration=success_estimator_timeout))

    # 4b.4 There is no special switch for unknown error response (estimator signals REFERENCE_MEASUREMENT_FAILURE)
    #      Instead the timeout will trigger giving the user an opportunity to notice the erroneous result in the GUI.

    # 5. Go down onto the object (relative in EE frame) - Godown
    control_sequence.append(ha.CartesianVelocityControlMode(down_twist,
                                             controller_name='GoDown',
                                             name="GoDown",
                                             reference_frame="EE"))

    # force threshold that if reached will trigger the closing of the hand
    force = np.array([0, 0, downward_force, 0, 0, 0])
    
    # 5b. Switch when force-torque sensor is triggered
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'softhand_close_1_0',
                                                 goal = force,
                                                 norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion = "THRESH_UPPER_BOUND",
                                                 goal_is_relative = '1',
                                                 frame_id = 'world'))

    # 5c. Switch to recovery if the cartesian velocity fails due to joint limits
    control_sequence.append(ha.RosTopicSwitch('GoDown', 'softhand_open_recovery_SurfaceGrasp', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 6. Call hand controller
    control_sequence.append(ha.BlockJointControlMode(name='softhand_close_1_0'))

    # 6b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch('softhand_close_1_0', 'GoUp_1', duration = hand_closing_time))


    return control_sequence