import numpy as np
import hatools.components as ha
from grasp_success_estimator import RESPONSES
from tf import transformations as tra
import math

def create_surface_grasp(chosen_object, handarm_params, pregrasp_transform, alternative_behavior=None):
    # Get robot specific params    
    joint_damping = handarm_params['joint_damping']
    
    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['SurfaceGrasp']:
        params = handarm_params['SurfaceGrasp'][object_type]
    else:
        params = handarm_params['SurfaceGrasp']['object']
    # Get params per phase

    # Approach phase
    high_joint_stiffness = params['high_joint_stiffness']
    downward_force = params['downward_force']
    down_speed = params['down_speed']

    # Grasping phase
    lift_time = params['corrective_lift_duration']
    up_speed = params['up_speed']
    hand_closing_time = params['hand_closing_duration']
    hand_preshaping_time = params['hand_preshaping_duration']   
    

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the EE frame
    down_twist = np.array([0, 0, down_speed, 0, 0, 0])
    # Slow Up speed is also positive because it is defined on the world frame
    up_twist = np.array([0, 0, up_speed, 0, 0, 0])

    # the 1 in softhand_close_1 represents a surface grasp. This way the strategy is encoded in the HA.
    mode_name_hand_closing = 'softhand_close_1_0'

    # assemble controller sequence
    control_sequence = []

    # 0a. Change arm mode - stiffen
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'joint_impedance', 
                                                        joint_stiffness = high_joint_stiffness, joint_damping = joint_damping))
    # 0b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoStiff', 'softhand_preshape_1_1', duration=1.0))

    #0. Trigger pre-shaping the hand (if there is a synergy). The first 1 in the name represents a surface grasp.
    control_sequence.append(ha.BlockJointControlMode(name = 'softhand_preshape_1_1'))
    
    # 0b. Time to trigger pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_1_1', 'PreGrasp', duration = hand_preshaping_time))


    #  # 2. Go above the object - PreGrasp
    # print('{}'.format(alternative_behavior))
    
    if alternative_behavior is not None and 'pre_approach' in alternative_behavior:

        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        goal_traj = alternative_behavior['pre_approach'].get_trajectory()

        print("Use alternative GOAL_TRAJ PreGrasp Dim", goal_traj.shape)
        control_sequence.append(ha.JointControlMode(goal_traj, name='PreGrasp', controller_name='GoAboveObject',
                                                    goal_is_relative='0',
                                                    v_max=np.array([0.005]),
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # 2b. Switch when hand reaches the goal configuration
        control_sequence.append(ha.JointConfigurationSwitch('PreGrasp', 'PrepareForMassMeasurement',
                                                            controller='GoAboveObject', epsilon=str(math.radians(7.))))

    else:
        # we can use the original motion
        control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform,
                                                                     name='PreGrasp',
                                                                     controller_name='GoAboveObject',
                                                                     goal_is_relative='0',
                                                                    #  v_max=pre_grasp_velocity,
                                                                      reference_frame = 'world'))

        # 1b. Switch when hand reaches the goal pose
        control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForMassMeasurement',
                                                   controller='GoAboveObject', epsilon='0.03'))    
    # 1b2. Switch when hand reaches the goal pose
    control_sequence.append(ha.RosTopicSwitch('PreGrasp', 'PrepareForMassMeasurement', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([2.])))
    
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

        # force threshold that if reached will trigger the closing of the hand
    force = np.array([0, 0, downward_force, 0, 0, 0])
      
    # 5. Go down onto the object - Godown
    if alternative_behavior is not None and 'go_down' in alternative_behavior:
        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        # Go down onto the object (joint controller + relative world frame motion)

        goal_traj = alternative_behavior['go_down'].get_trajectory()

        print("Use alternative GOAL_TRAJ GoDown Dim:", goal_traj.shape)  # TODO Check if the dimensions are correct and the via points are as expected
        control_sequence.append(ha.JointControlMode(goal_traj, name='GoDown', controller_name='GoDown',
                                                    goal_is_relative='0',
                                                    # v_max=go_down_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Relative motion that ensures that the actual force/torque threshold is reached
        
        control_sequence.append(ha.CartesianVelocityControlMode(down_twist,
                                                                controller_name='GoDownFurther',
                                                                name="GoDownFurther",
                                                                reference_frame="EE"))

        # Switch when goal is reached to trigger the relative go down further motion
        control_sequence.append(ha.JointConfigurationSwitch('GoDown', 'GoDownFurther', controller='GoDown',
                                                            epsilon=str(math.radians(7.))))

        # Force/Torque switch for the additional relative go down further
        control_sequence.append(ha.ForceTorqueSwitch('GoDownFurther',
                                                     mode_name_hand_closing,
                                                     goal=force,
                                                     norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion="THRESH_UPPER_BOUND",
                                                     goal_is_relative='1',
                                                     frame_id='world',
                                                     port='2'))

    else:
        # Go down onto the object (relative in world frame)
        control_sequence.append(ha.CartesianVelocityControlMode(down_twist,
                                                                controller_name='GoDown',
                                                                name="GoDown",
                                                                reference_frame="EE"))

    # 4b. Switch when force-torque sensor is triggered
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'LiftHand',
                                                 goal = force,
                                                 norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion = "THRESH_UPPER_BOUND",
                                                 goal_is_relative = '1',
                                                 frame_id = 'world'))

    # 4c. Switch to recovery if the cartesian velocity fails due to joint limits
    control_sequence.append(ha.RosTopicSwitch('GoDown', 'softhand_open_recovery_SurfaceGrasp', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 5. Lift upwards so the hand can inflate
    control_sequence.append(
        ha.CartesianVelocityControlMode(up_twist, controller_name='CorrectiveLift', name="LiftHand",
                                             reference_frame="world"))



    # 5b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('LiftHand', mode_name_hand_closing, duration=lift_time))

    # 5c. Change arm mode - soften
    # control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoSoft', mode_id = 'joint_impedance', 
    #                                                     joint_stiffness = soft_joint_stiffness, joint_damping = joint_damping))
    # # 5d. We switch after a short time 
    # control_sequence.append(ha.TimeSwitch('GoSoft', mode_name_hand_closing, duration=1.0))

    # 6. Call hand controller
    control_sequence.append(ha.BlockJointControlMode(name =mode_name_hand_closing))
   
    # 6b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'GoUp_1', duration = hand_closing_time))

    return control_sequence


# ================================================================================================
def create_wall_grasp(chosen_object, wall_frame, handarm_params, pregrasp_transform, grasp_type = 'WallGrasp', alternative_behavior=None):
    # Get robot specific params    
    joint_damping = handarm_params['joint_damping']

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params[grasp_type]:
        params = handarm_params[grasp_type][object_type]
    else:
        params = handarm_params[grasp_type]['object']

    high_joint_stiffness = params['high_joint_stiffness']
    low_joint_stiffness = params['low_joint_stiffness']

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
    hand_preshaping_time = params['hand_preshaping_duration']

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

    # 0 trigger pre-shaping the hand (if there is a synergy). The 2 in the name represents a wall grasp.
    control_sequence.append(ha.BlockJointControlMode(name='softhand_preshape_2_1'))
    
    # 0b. Time for pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_2_1', 'GoStiff', duration=hand_preshaping_time)) 

    # 0a. Change arm mode - stiffen
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoStiff', mode_id = 'joint_impedance', 
                                                        joint_stiffness = high_joint_stiffness, joint_damping = joint_damping))
    # 0b. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoStiff', 'PreGrasp', duration=1.0))

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'PreGrasp', reference_frame = 'world'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForMassMeasurement', controller = 'GoAboveObject', epsilon = '0.01'))

    # 1b2. Switch when hand reaches the goal pose
    control_sequence.append(ha.RosTopicSwitch('PreGrasp', 'PrepareForMassMeasurement', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([2.])))
    
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
    control_sequence.append(ha.kukaChangeModeControlMode(name = 'GoSoft', mode_id = 'joint_impedance', 
                                                        joint_stiffness = low_joint_stiffness, joint_damping = joint_damping))
    # 6e. We switch after a short time 
    control_sequence.append(ha.TimeSwitch('GoSoft', 'SlideBackFromWall', duration=0))

    # 7. Go back a bit to allow the hand to inflate
    control_sequence.append(
        ha.CartesianVelocityControlMode(pre_grasp_twist, controller_name='SlideBackFromWall',
                                             name="SlideBackFromWall", reference_frame="EE"))
    # The 2 in softhand_close_2 represents a wall grasp. This way the strategy is encoded in the HA.
    # The 0 encodes the synergy id
    mode_name_hand_closing = 'softhand_close_2_0'

    # 7b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('SlideBackFromWall', mode_name_hand_closing, duration=1.4))
    
    # 8. Close the hand
    control_sequence.append(ha.BlockJointControlMode(name  = mode_name_hand_closing))
    
    # 8b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'PostGraspRotate', duration=hand_closing_time))

    # 9. Rotate a bit to roll the object in the hand
    control_sequence.append(
        ha.CartesianVelocityControlMode(post_grasp_twist, controller_name='PostGraspRotate',
                                             name="PostGraspRotate", reference_frame="EE"))
    # 9b. We switch after a short time
    control_sequence.append(ha.TimeSwitch('PostGraspRotate', 'GoUp_1', duration=post_grasp_rotate_time))
    
    return control_sequence

# ================================================================================================
def create_corner_grasp(chosen_object, corner_frame_alpha_zero, handarm_params, pregrasp_transform):

    return create_wall_grasp(chosen_object, corner_frame_alpha_zero, handarm_params, pregrasp_transform, grasp_type = 'CornerGrasp')
