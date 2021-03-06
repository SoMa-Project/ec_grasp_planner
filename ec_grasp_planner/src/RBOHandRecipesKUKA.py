import numpy as np
import hatools.components as ha
from grasp_success_estimator import RESPONSES


def create_surface_grasp(chosen_object, handarm_params, pregrasp_transform, alternative_behavior=None):

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['SurfaceGrasp']:
        params = handarm_params['SurfaceGrasp'][object_type]
    else:
        params = handarm_params['SurfaceGrasp']['object']
    # Get params per phase

    # Approach phase
    downward_force = params['downward_force']
    down_speed = params['down_speed']

    # Grasping phase
    lift_time = params['corrective_lift_duration']
    up_speed = params['up_speed']
    hand_closing_time = params['hand_closing_duration']
    hand_synergy = params['hand_closing_synergy']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the EE frame
    down_twist = np.array([0, 0, down_speed, 0, 0, 0])
    # Slow Up speed is also positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])

    # assemble controller sequence
    control_sequence = []

    # 0. Trigger pre-shaping the hand (if there is a synergy). The first 1 in the name represents a surface grasp.
    control_sequence.append(ha.BlockJointControlMode(name = 'softhand_preshape_1_1'))
    
    # 0b. Time to trigger pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_1_1', 'PreGrasp', duration = hand_closing_time))

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'PreGrasp'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForMassMeasurement', controller = 'GoAboveObject', epsilon = '0.01'))
    
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
                                                 'LiftHand',
                                                 goal = force,
                                                 norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion = "THRESH_UPPER_BOUND",
                                                 goal_is_relative = '1',
                                                 frame_id = 'world'))

    # 5. Lift upwards so the hand can inflate
    control_sequence.append(
        ha.CartesianVelocityControlMode(up_twist, controller_name='CorrectiveLift', name="LiftHand",
                                             reference_frame="world"))

    # the 1 in softhand_close_1 represents a surface grasp. This way the strategy is encoded in the HA.
    mode_name_hand_closing = 'softhand_close_1_0'

    # 5b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('LiftHand', mode_name_hand_closing, duration=lift_time))

    # 6. Call hand controller
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name = mode_name_hand_closing, synergy = hand_synergy))
   
    # 6b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'GoUp_1', duration = hand_closing_time))

    return control_sequence


# ================================================================================================
def create_wall_grasp(chosen_object, wall_frame, handarm_params, pregrasp_transform, alternative_behavior=None):

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['WallGrasp']:
        params = handarm_params['WallGrasp'][object_type]
    else:
        params = handarm_params['WallGrasp']['object']

    # Get params per phase

    # Approach phase
    downward_force = params['downward_force']
    down_speed = params['down_speed']

    lift_time = params['corrective_lift_duration']
    up_speed = params['up_speed']

    wall_force = params['wall_force']
    slide_speed = params['slide_speed']

    # Grasping phase
    pre_grasp_twist = params['pre_grasp_twist']
    pre_grasp_rotate_time = params['pre_grasp_rotation_duration']
    hand_closing_time = params['hand_closing_duration']
    hand_synergy = params['hand_closing_synergy']

    # Post-grasping phase
    post_grasp_twist = params['post_grasp_twist']
    post_grasp_rotate_time = params['post_grasp_rotation_duration']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the twists to use TRIK controller with

    # Down speed is negative because it is defined on the world frame
    down_twist = np.array([0, 0, -down_speed, 0, 0, 0])
    # Slow Up speed is positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])
    # Slide twist is positive because it is defined on the EE frame
    slide_twist = np.array([0, 0, slide_speed, 0, 0, 0])

    control_sequence = []

    # 0 trigger pre-shaping the hand (if there is a synergy). The 2 in the name represents a wall grasp.
    control_sequence.append(ha.BlockJointControlMode(name='softhand_preshape_2_1'))
    
    # 0b. Time for pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_2_1', 'PreGrasp', duration=hand_closing_time)) 

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'PreGrasp'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForMassMeasurement', controller = 'GoAboveObject', epsilon = '0.01'))
    
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

    # 5. Lift upwards so the hand doesn't slide on table surface
    control_sequence.append(
        ha.CartesianVelocityControlMode(up_twist, controller_name='Lift1', name="LiftHand",
                                             reference_frame="world"))

    # 5b. We switch after a short time as this allows us to do a small, precise lift motion
    control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=lift_time))

    # 6. Go towards the wall to slide object to wall
    control_sequence.append(
        ha.CartesianVelocityControlMode(slide_twist, controller_name='SlideToWall',
                                             name="SlideToWall", reference_frame="EE"))

    # 6b. Switch when the f/t sensor is triggered with normal force from wall
    force = np.array([0, 0, wall_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', 'SlideBackFromWall', 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame))

    # 7. Go back a bit to allow the hand to inflate
    control_sequence.append(
        ha.CartesianVelocityControlMode(pre_grasp_twist, controller_name='SlideBackFromWall',
                                             name="SlideBackFromWall", reference_frame="EE"))
    # The 2 in softhand_close_2 represents a wall grasp. This way the strategy is encoded in the HA.
    # The 0 encodes the synergy id
    mode_name_hand_closing = 'softhand_close_2_0'

    # 7b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('SlideBackFromWall', mode_name_hand_closing, duration=pre_grasp_rotate_time))
    

    # 8. Maintain contact while closing the hand
    
    # Call general hand controller
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = mode_name_hand_closing, synergy = hand_synergy))
    

    # 8b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'PostGraspRotate', duration=hand_closing_time))

    # 9. Rotate a bit to roll the object in the hand
    control_sequence.append(
        ha.CartesianVelocityControlMode(post_grasp_twist, controller_name='PostGraspRotate',
                                             name="PostGraspRotate", reference_frame="EE"))
    # 9b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('PostGraspRotate', 'GoUp_1', duration=post_grasp_rotate_time))
    
    return control_sequence