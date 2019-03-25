import numpy as np
import hatools.components as ha
import math
from tf import transformations as tra
from grasp_success_estimator import RESPONSES


def create_surface_grasp(chosen_object, handarm_params, pregrasp_transform, alternative_behavior=None):

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['SurfaceGrasp']:
        params = handarm_params['SurfaceGrasp'][object_type]
    else:
        params = handarm_params['SurfaceGrasp']['object']

    # ############################ #
    # Get params per phase
    # ############################ #

    # -- Approach phase --

    # Move to initial joint config that is better than vision config w.r.t. proximity to joint limits
    # TODO get rid of this workaround since we actually have joint space planning capabilities
    initial_jointConf = params['initial_goal']

    # Pre-Grasp modification (re-introducing hacky version for our WAM)
    # The grasp frame is symmetrical - check which side is nicer to reach
    # TODO: check if we can remove this part now, because of feasibilty check
    param_pre_grasp = params['pre_approach_transform']

    # goal_ is object_pose.dot(params['hand_transform'])
    goal_ = pregrasp_transform.dot(tra.inverse_matrix(param_pre_grasp))

    # hacky check (if feasibility is on, this is already ensured, otherwise we need this for the basic heuristic)
    if goal_[0][0] < 0:
        zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
        goal_ = goal_.dot(zflip_transform)
        pregrasp_transform = goal_.dot(param_pre_grasp)
    
    # Pre-Grasp modification end TODO get rid of it

    pre_grasp_velocity = params['pre_approach_velocity']
    pre_grasp_joint_velocity = params['pre_approach_joint_velocity']

    # force threshold that if reached will trigger the closing of the hand
    downward_force_thresh = np.array([0, 0, params['downward_force'], 0, 0, 0])
    
    down_dist = params['down_dist']
    down_dir = tra.translation_matrix([0, 0, -down_dist])

    go_down_velocity = params['go_down_velocity']
    go_down_joint_velocity = params['go_down_joint_velocity']

    # -- Grasping phase --

    # the 1 in softhand_close_1 represents a surface grasp. This way the strategy is encoded in the HA.
    hand_synergy = params['hand_closing_synergy']
    mode_name_hand_closing = 'softhand_close_1_' + str(hand_synergy)

    hand_closing_duration = params['hand_closing_duration']

    # -- Post-grasping phase --
    post_grasp_transform = params['post_grasp_transform']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the frames to visualize with RViz # TODO bring that feature back to recipes
    # rviz_frames = [object_frame, goal_, pre_grasp_pose]

    # ############################ #
    # Assemble controller sequence
    # ############################ #

    control_sequence = []

    # 0. Initial position above ifco
    control_sequence.append(
        ha.JointControlMode(initial_jointConf, name='InitialJointConfig', controller_name='initialJointCtrl'))

    # 0b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('InitialJointConfig', 'softhand_preshape_1_1',
                                                        controller='initialJointCtrl', epsilon=str(math.radians(7.))))

    # 1. Trigger pre-shaping the hand (if there is a synergy). The first 1 in the name represents a surface grasp.
    control_sequence.append(ha.BlockJointControlMode(name='softhand_preshape_1_1'))

    # 1b. Time to trigger pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_1_1', 'PreGrasp', duration=0.2))

    # 2. Go above the object - PreGrasp
    if alternative_behavior is not None and 'pre_grasp' in alternative_behavior:

        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        goal_traj = alternative_behavior['pre_grasp'].get_trajectory()

        print("Use alternative GOAL_TRAJ PreGrasp Dim", goal_traj.shape)
        control_sequence.append(ha.JointControlMode(goal_traj, name='PreGrasp', controller_name='GoAboveObject',
                                                    goal_is_relative='0',
                                                    v_max=pre_grasp_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # 2b. Switch when hand reaches the goal configuration
        control_sequence.append(ha.JointConfigurationSwitch('PreGrasp', 'PrepareForReferenceMassMeasurement',
                                                            controller='GoAboveObject', epsilon=str(math.radians(7.))))

    else:
        # we can use the original motion
        control_sequence.append(ha.InterpolatedHTransformControlMode(pregrasp_transform,
                                                                     name='PreGrasp',
                                                                     controller_name='GoAboveObject',
                                                                     goal_is_relative='0',
                                                                     v_max=pre_grasp_velocity))

        # 1b. Switch when hand reaches the goal pose
        control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForReferenceMassMeasurement',
                                                   controller='GoAboveObject', epsilon='0.01'))

    # 3. Hold current joint config 
    control_sequence.append(ha.BlockJointControlMode(name='PrepareForReferenceMassMeasurement'))

    # 3b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('PrepareForReferenceMassMeasurement', 'ReferenceMassMeasurement',
                                          duration=0.5))

    # 4. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))

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

    # 5. Go down onto the object - Godown
    if alternative_behavior is not None and 'go_down' in alternative_behavior:
        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        # Go down onto the object (joint controller + relative world frame motion)

        goal_traj = alternative_behavior['go_down'].get_trajectory()

        print("Use alternative GOAL_TRAJ GoDown Dim:", goal_traj.shape)  # TODO Check if the dimensions are correct and the via points are as expected
        control_sequence.append(ha.JointControlMode(goal_traj, name='GoDown', controller_name='GoDown',
                                                    goal_is_relative='0',
                                                    v_max=go_down_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Relative motion that ensures that the actual force/torque threshold is reached
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(tra.translation_matrix([0, 0, -10]),
                                                 controller_name='GoDownFurther',
                                                 goal_is_relative='1',
                                                 name="GoDownFurther",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

        # Switch when goal is reached to trigger the relative go down further motion
        control_sequence.append(ha.JointConfigurationSwitch('GoDown', 'GoDownFurther', controller='GoDown',
                                                            epsilon=str(math.radians(7.))))

        # Force/Torque switch for the additional relative go down further
        control_sequence.append(ha.ForceTorqueSwitch('GoDownFurther',
                                                     mode_name_hand_closing,
                                                     goal=downward_force_thresh,
                                                     norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion="THRESH_UPPER_BOUND",
                                                     goal_is_relative='1',
                                                     frame_id='world',
                                                     port='2'))

    else:
        # Go down onto the object (relative in world frame)
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(down_dir,
                                                 controller_name='GoDown',
                                                 goal_is_relative='1',
                                                 name="GoDown",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

    # 5b. Force/Torque switch for GoDown (in both cases joint trajectory or op-space control)
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 mode_name_hand_closing,
                                                 goal=downward_force_thresh,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))

    # 6. Call hand controller (maintain the position during grasp)
    
    desired_displacement = np.array([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]])

    force_gradient = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.005],
                               [0.0, 0.0, 0.0, 1.0]])

    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    control_sequence.append(ha.HandControlMode_ForceHT(name=mode_name_hand_closing, synergy=hand_synergy,
                                                       desired_displacement=desired_displacement,
                                                       force_gradient=force_gradient,
                                                       desired_force_dimension=desired_force_dimension))

    # 6b. Switch when hand closing time ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'PostGraspRotate', duration=hand_closing_duration))

    # 7. Rotate hand after closing and before lifting it up relative to current hand pose
    control_sequence.append(ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate',
                                                     name='PostGraspRotate', goal_is_relative='1'))

    # 7b. Switch when hand rotated for a bit
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp_1', controller='PostGraspRotate',
                                               epsilon='0.01', goal_is_relative='1', reference_frame='EE'))

    return control_sequence  # TODO what about rviz_frames?


# ================================================================================================
def create_wall_grasp(chosen_object, wall_frame, handarm_params, pregrasp_transform, alternative_behavior=None):

    # the pre-approach (pregrasp_transform) pose should be:
    # - floating above and behind the object,
    # - fingers pointing downwards
    # - palm facing the object and wall

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['WallGrasp']:
        params = handarm_params['WallGrasp'][object_type]
    else:
        params = handarm_params['WallGrasp']['object']

    # ############################ #
    # Get params per phase
    # ############################ #

    # -- Approach phase --

    initial_jointConf = params['initial_goal']

    pre_approach_transform = pregrasp_transform
    pre_grasp_joint_velocity = params['max_joint_velocity']

    # ---- GoDown ----
    # Force threshold for guarded GoDown motion
    downward_force_thresh = np.array([0, 0, params['downward_force'], 0, 0, 0])

    down_dist = params['down_dist']
    down_dir = tra.translation_matrix([0, 0, -down_dist])

    go_down_velocity = params['go_down_velocity']
    go_down_joint_velocity = params['max_joint_velocity']

    # ---- LiftHand ----
    lift_dist = params['corrective_lift_dist']
    lift_dir = tra.translation_matrix([0, 0, lift_dist])
    lift_hand_joint_velocity = params['max_joint_velocity']

    # ---- SlideToWall
    wall_force = params['wall_force']
    wall_force_threshold = np.array([0, 0, wall_force, 0, 0, 0])

    # TODO sliding_distance should be computed from wall and hand frame.
    sliding_dist = params['sliding_dist']
    # slide direction is given by the normal of the wall
    wall_dir = tra.translation_matrix([0, 0, -sliding_dist])
    wall_dir[:3, 3] = wall_frame[:3, :3].dot(wall_dir[:3, 3])

    slide_velocity = params['slide_velocity']
    slide_joint_velocity = params['slide_joint_velocity']

    # -- Grasping phase --

    # The 2 in softhand_close_2 represents a wall grasp. This way the strategy is encoded in the HA.
    hand_synergy = params['hand_closing_synergy']
    mode_name_hand_closing = 'softhand_close_2_' + str(hand_synergy)

    hand_closing_duration = params['hand_closing_duration']

    # -- Post-grasping phase --
    post_grasp_transform = params['post_grasp_transform']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the frames to visualize with Rviz # TODO bring that feature back to recipes
    # global rviz_frames
    # rviz_frames = []
    # rviz_frames.append(pre_approach_transform)
    #
    # transformation between hand and EC frame (which is positioned like object and oriented like wall) at grasp time
    # hand_transform = params['hand_transform']
    #
    # this is the EC frame. It is positioned like object and oriented to the wall
    # ec_frame = np.copy(wall_frame)
    # ec_frame[:3, 3] = tra.translation_from_matrix(object_frame)
    # apply hand transformation
    # ec_frame = (ec_frame.dot(hand_transform))
    #
    # rviz_frames.append(ec_frame)
    # rviz_frames.append(wall_frame)

    # ############################ #
    # Assemble controller sequence
    # ############################ #

    control_sequence = []

    # 0. Initial position above ifco
    control_sequence.append(
        ha.JointControlMode(initial_jointConf, name='InitialJointConfig', controller_name='initialJointCtrl'))

    # 0b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('InitialJointConfig', 'softhand_preshape_2_1',
                                                        controller='initialJointCtrl', epsilon=str(math.radians(7.))))

    # 1. trigger pre-shaping the hand (if there is a synergy). The 2 in the name represents a wall grasp.
    control_sequence.append(ha.BlockJointControlMode(name='softhand_preshape_2_1'))

    # 1b. Time for pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_2_1', 'PreGrasp', duration=0.5))

    # 2. Go above the object - Pregrasp
    if alternative_behavior is not None and 'pre_grasp' in alternative_behavior:

        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        goal_traj = alternative_behavior['pre_grasp'].get_trajectory()

        print("Use alternative GOAL_TRAJ PreGrasp Dim", goal_traj.shape)
        control_sequence.append(ha.JointControlMode(goal_traj, name='PreGrasp', controller_name='GoAboveObject',
                                                    goal_is_relative='0',
                                                    v_max=pre_grasp_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # 2b. Switch when hand reaches the goal configuration
        control_sequence.append(ha.JointConfigurationSwitch('PreGrasp', 'PrepareForReferenceMassMeasurement',
                                                            controller='GoAboveObject', epsilon=str(math.radians(7.))))

    else:
        # we can use the original motion
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(pre_approach_transform, controller_name='GoAboveObject',
                                                 goal_is_relative='0', name="PreGrasp"))

        # 2b. Switch when hand reaches the goal pose
        control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForReferenceMassMeasurement',
                                                   controller='GoAboveObject', epsilon='0.01'))

    # 3. Hold current joint config 
    control_sequence.append(ha.BlockJointControlMode(name='PrepareForReferenceMassMeasurement'))

    # 3b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('PrepareForReferenceMassMeasurement', 'ReferenceMassMeasurement',
                                          duration=0.5))

    # 4. Reference mass measurement with empty hand (TODO can this be replaced by offline calibration?)
    control_sequence.append(ha.BlockJointControlMode(name='ReferenceMassMeasurement'))

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

    # 5. Go down onto the object/table, in world frame

    if alternative_behavior is not None and 'go_down' in alternative_behavior:
        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        # Go down onto the object (joint controller + relative world frame motion)

        goal_traj = alternative_behavior['go_down'].get_trajectory()

        print("Use alternative GOAL_TRAJ GoDown Dim:", goal_traj.shape)  # TODO remove (only for debugging)
        control_sequence.append(ha.JointControlMode(goal_traj, name='GoDown', controller_name='GoDown',
                                                    goal_is_relative='0',
                                                    v_max=go_down_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Relative motion that ensures that the actual force/torque threshold is reached
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(tra.translation_matrix([0, 0, -10]),
                                                 controller_name='GoDownFurther',
                                                 goal_is_relative='1',
                                                 name="GoDownFurther",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

        # Switch when goal is reached to trigger the relative go down further motion
        control_sequence.append(ha.JointConfigurationSwitch('GoDown', 'GoDownFurther', controller='GoDown',
                                                            epsilon=str(math.radians(7.))))

        # Force/Torque switch for the additional relative go down further
        control_sequence.append(ha.ForceTorqueSwitch('GoDownFurther',
                                                     'LiftHand',
                                                     goal=downward_force_thresh,
                                                     norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion="THRESH_UPPER_BOUND",
                                                     goal_is_relative='1',
                                                     frame_id='world',
                                                     port='2'))
    else:
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(down_dir,
                                                 controller_name='GoDown',
                                                 goal_is_relative='1',
                                                 name="GoDown",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

    # 5b. Switch when force threshold is exceeded (in both cases joint trajectory or op-space control)
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'LiftHand',
                                                 goal=downward_force_thresh,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))

    # 6. Lift upwards so the hand doesn't slide directly on table surface
    lift_duration = 0.2  # timeout for the TimeSwitch-hack
    if alternative_behavior is not None and 'lift_hand' in alternative_behavior:

        goal_traj = alternative_behavior['lift_hand'].get_trajectory()
        print("Use alternative GOAL_TRAJ LiftHand Dim:", goal_traj.shape)  # TODO remove (only for debugging)
        control_sequence.append(ha.JointControlMode(goal_traj, name='LiftHand', controller_name='Lift1',
                                                    goal_is_relative='0',
                                                    v_max=lift_hand_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Switch when goal is reached to trigger the relative go down further motion
        control_sequence.append(ha.JointConfigurationSwitch('LiftHand', 'SlideToWall', controller='Lift1',
                                                            epsilon=str(math.radians(7.))))

        # We also switch after a short time as this allows us to do a small, precise lift motion, in case the
        # JointConfigurationSwitch above does not trigger.
        # TODO partners: this can be removed if your robot is able to do small motions precisely
        control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=lift_duration))

    else:
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(lift_dir, controller_name='Lift1', goal_is_relative='1',
                                                 name="LiftHand", reference_frame="world"))

        # 6b. We switch after a short time as this allows us to do a small, precise lift motion
        # TODO partners: this can be replaced by a frame pose switch if your robot is able to do small motions precisely
        control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=lift_duration))

    # 7. Go towards the wall to slide object to wall
    if alternative_behavior is not None and 'slide_to_wall' in alternative_behavior:

        goal_traj = alternative_behavior['slide_to_wall'].get_trajectory()
        print("Use alternative GOAL_TRAJ SlideToWall Dim:", goal_traj.shape)  # TODO remove (only for debugging)

        # Sliding motion in joint space
        control_sequence.append(ha.JointControlMode(goal_traj, name='SlideToWall', controller_name='SlideToWall',
                                                    goal_is_relative='0',
                                                    v_max=slide_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Relative motion that ensures that the actual force/torque threshold is reached (world space)
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(wall_dir,
                                                 controller_name='SlideFurther',
                                                 goal_is_relative='1',
                                                 name="SlideFurther",
                                                 reference_frame="world",
                                                 v_max=slide_velocity))

        # Switch when goal is reached to trigger the relative slide further motion
        control_sequence.append(ha.JointConfigurationSwitch('SlideToWall', 'SlideFurther', controller='SlideToWall',
                                                            epsilon=str(math.radians(7.))))

        # Force/Torque switch for the additional relative slide further
        control_sequence.append(ha.ForceTorqueSwitch('SlideFurther',
                                                     mode_name_hand_closing,
                                                     name='ForceSwitch',
                                                     goal=wall_force_threshold,
                                                     norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion="THRESH_UPPER_BOUND",
                                                     goal_is_relative='1',
                                                     frame_id='world',
                                                     frame=wall_frame,
                                                     port='2'))

    else:
        # Sliding motion (entirely world space controlled)
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(wall_dir, controller_name='SlideToWall', goal_is_relative='1',
                                                 name="SlideToWall", reference_frame="world",
                                                 v_max=slide_velocity))

    # 7b. Switch when the f/t sensor is triggered with normal force from wall
    #     (in both cases joint trajectory or op-space control)
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', mode_name_hand_closing, name='ForceSwitch',
                                                 goal=wall_force_threshold,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame, port='2'))

    # 8. Maintain contact while closing the hand
    # apply force on object while closing the hand
    # TODO arne: validate these values
    desired_displacement = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    force_gradient = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    control_sequence.append(ha.HandControlMode_ForceHT(name=mode_name_hand_closing, synergy=hand_synergy,
                                                       desired_displacement=desired_displacement,
                                                       force_gradient=force_gradient,
                                                       desired_force_dimension=desired_force_dimension))

    # 8b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'PostGraspRotate', duration=hand_closing_duration))

    # 9. Move hand after closing and before lifting it up to roll the object in the hand
    # relative to current hand pose
    control_sequence.append(
        ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate',
                                 goal_is_relative='1', ))

    # 9b. Switch when hand reaches post grasp pose
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp_1', controller='PostGraspRotate',
                                               epsilon='0.01', goal_is_relative='1', reference_frame='EE'))

    return control_sequence  # TODO what about rviz_frames?


# ================================================================================================
def create_corner_grasp(chosen_object, corner_frame_alpha_zero, handarm_params, pregrasp_transform,
                        alternative_behavior=None):

    # the pre-approach pose should be:
    # - floating above and behind the object,
    # - fingers pointing downwards
    # - palm facing the object and wall

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params['CornerGrasp']:
        params = handarm_params['CornerGrasp'][object_type]
    else:
        params = handarm_params['CornerGrasp']['object']

    # ############################ #
    # Get params per phase
    # ############################ #

    # -- Approach phase --

    # initial configuration above IFCO. Should be easy to go from here to pregrasp pose
    initial_jointConf = params['initial_goal']

    pre_approach_transform = pregrasp_transform
    pre_grasp_joint_velocity = params['max_joint_velocity']

    # ---- GoDown ----
    # Force threshold for guarded GoDown motion
    downward_force_thresh = np.array([0, 0, params['downward_force'], 0, 0, 0])

    down_dist = params['down_dist']
    down_dir = tra.translation_matrix([0, 0, -down_dist])

    go_down_velocity = params['go_down_velocity']
    go_down_joint_velocity = params['max_joint_velocity']

    # ---- LiftHand ----
    lift_dist = params['corrective_lift_dist']
    lift_dir = tra.translation_matrix([0, 0, lift_dist])
    lift_hand_joint_velocity = params['max_joint_velocity']

    # ---- SlideToWall
    wall_force = params['wall_force']
    wall_force_threshold = np.array([0, 0, wall_force, 0, 0, 0])

    # TODO sliding_distance should be computed from wall and hand frame.
    sliding_dist = params['sliding_dist']
    wall_dir = tra.translation_matrix([0, 0, -sliding_dist])
    # slide direction is given by the corner_frame_alpha_zero
    wall_dir[:3, 3] = corner_frame_alpha_zero[:3, :3].dot(wall_dir[:3, 3])

    slide_velocity = params['slide_velocity']
    slide_joint_velocity = params['slide_joint_velocity']

    # -- Grasping phase --

    # The 4 in softhand_close_4 represents a corner grasp. This way the strategy is encoded in the HA.
    hand_synergy = params['hand_closing_synergy']
    mode_name_hand_closing = 'softhand_close_4_' + str(hand_synergy)

    hand_closing_duration = params['hand_closing_duration']

    # -- Post-grasping phase --
    post_grasp_transform = params['post_grasp_transform']

    success_estimator_timeout = handarm_params['success_estimator_timeout']

    # Set the debug frames to visualize with Rviz # TODO bring that feature back to recipes
    # global rviz_frames
    # rviz_frames = []
    # rviz_frames.append(pre_approach_transform)
    # rviz_frames.append(ec_frame)
    # rviz_frames.append(corner_frame)
    # rviz_frames.append(corner_frame_alpha_zero)

    # ############################ #
    # Assemble controller sequence
    # ############################ #

    control_sequence = []

    # 0. initial position above ifco
    control_sequence.append(
        ha.JointControlMode(initial_jointConf, name='InitialJointConfig', controller_name='initialJointCtrl'))

    # 0b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('InitialJointConfig', 'softhand_preshape_4_1',
                                                        controller='initialJointCtrl', epsilon=str(math.radians(7.))))

    # 1. trigger pre-shaping the hand (if there is a synergy). The 4 in the name represents a corner grasp.
    control_sequence.append(ha.BlockJointControlMode(name='softhand_preshape_4_1'))

    # 1b. Time for pre-shape
    control_sequence.append(ha.TimeSwitch('softhand_preshape_4_1', 'PreGrasp', duration=0.5))  # time for pre-shape

    # 2. Go above the object - Pregrasp
    if alternative_behavior is not None and 'pre_grasp' in alternative_behavior:
        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        goal_traj = alternative_behavior['pre_grasp'].get_trajectory()

        print("Use alternative GOAL_TRAJ PreGrasp Dim", goal_traj.shape)
        control_sequence.append(ha.JointControlMode(goal_traj, name='PreGrasp', controller_name='GoAboveObject',
                                                    goal_is_relative='0',
                                                    v_max=pre_grasp_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # 2b. Switch when hand reaches the goal configuration
        control_sequence.append(ha.JointConfigurationSwitch('PreGrasp', 'PrepareForReferenceMassMeasurement',
                                                            controller='GoAboveObject', epsilon=str(math.radians(7.))))

    else:
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(pre_approach_transform, controller_name='GoAboveObject',
                                                 goal_is_relative='0', name="PreGrasp"))

        # 2b. Switch when hand reaches the goal pose
        control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'PrepareForReferenceMassMeasurement',
                                                   controller='GoAboveObject', epsilon='0.01'))

    # 3. Hold current joint config
    control_sequence.append(ha.BlockJointControlMode(name='PrepareForReferenceMassMeasurement'))

    # 3b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('PrepareForReferenceMassMeasurement', 'ReferenceMassMeasurement',
                                          duration=0.5))

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

    # 5. Go down onto the object/table, in world frame

    if alternative_behavior is not None and 'go_down' in alternative_behavior:
        # we can not use the initially generated plan, but have to include the result of the feasibility checks
        # Go down onto the object (joint controller + relative world frame motion)

        goal_traj = alternative_behavior['go_down'].get_trajectory()

        print("Use alternative GOAL_TRAJ GoDown Dim:", goal_traj.shape)  # TODO remove (only for debugging)
        control_sequence.append(ha.JointControlMode(goal_traj, name='GoDown', controller_name='GoDown',
                                                    goal_is_relative='0',
                                                    v_max=go_down_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Relative motion that ensures that the actual force/torque threshold is reached
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(tra.translation_matrix([0, 0, -10]),
                                                 controller_name='GoDownFurther',
                                                 goal_is_relative='1',
                                                 name="GoDownFurther",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

        # Switch when goal is reached to trigger the relative go down further motion
        control_sequence.append(ha.JointConfigurationSwitch('GoDown', 'GoDownFurther', controller='GoDown',
                                                            epsilon=str(math.radians(7.))))

        # Force/Torque switch for the additional relative go down further
        control_sequence.append(ha.ForceTorqueSwitch('GoDownFurther',
                                                     'LiftHand',
                                                     goal=downward_force_thresh,
                                                     norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion="THRESH_UPPER_BOUND",
                                                     goal_is_relative='1',
                                                     frame_id='world',
                                                     port='2'))
    else:
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(down_dir,
                                                 controller_name='GoDown',
                                                 goal_is_relative='1',
                                                 name="GoDown",
                                                 reference_frame="world",
                                                 v_max=go_down_velocity))

    # 5b. Switch when force threshold is exceeded
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'LiftHand',
                                                 goal=downward_force_thresh,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))

    # 6. Lift upwards so the hand doesn't slide directly on table surface
    lift_duration = 0.2  # timeout for the TimeSwitch-hack
    if alternative_behavior is not None and 'lift_hand' in alternative_behavior:

        goal_traj = alternative_behavior['lift_hand'].get_trajectory()
        print("Use alternative GOAL_TRAJ LiftHand Dim:", goal_traj.shape)  # TODO remove (only for debugging)
        control_sequence.append(ha.JointControlMode(goal_traj, name='LiftHand', controller_name='Lift1',
                                                    goal_is_relative='0',
                                                    v_max=lift_hand_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Switch when goal is reached to trigger the relative go down further motion
        control_sequence.append(ha.JointConfigurationSwitch('LiftHand', 'SlideToWall', controller='Lift1',
                                                            epsilon=str(math.radians(7.))))

        # We also switch after a short time as this allows us to do a small, precise lift motion, in case the
        # JointConfigurationSwitch above does not trigger.
        # TODO partners: this can be removed if your robot is able to do small motions precisely
        control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=lift_duration))

    else:
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(lift_dir, controller_name='Lift1', goal_is_relative='1',
                                                 name="LiftHand", reference_frame="world"))

        # 6b. We switch after a short time as this allows us to do a small, precise lift motion
        # TODO partners: this can be replaced by a frame pose switch if your robot is able to do small motions precisely
        control_sequence.append(ha.TimeSwitch('LiftHand', 'SlideToWall', duration=lift_duration))

    # 7. Go towards the wall to slide object to wall
    if alternative_behavior is not None and 'slide_to_wall' in alternative_behavior:

        goal_traj = alternative_behavior['slide_to_wall'].get_trajectory()
        print("Use alternative GOAL_TRAJ SlideToWall Dim:", goal_traj.shape)  # TODO remove (only for debugging)

        # Sliding motion in joint space
        control_sequence.append(ha.JointControlMode(goal_traj, name='SlideToWall', controller_name='SlideToWall',
                                                    goal_is_relative='0',
                                                    v_max=slide_joint_velocity,
                                                    # for the close trajectory points linear interpolation works best.
                                                    interpolation_type='linear'))

        # Relative motion that ensures that the actual force/torque threshold is reached (world space)
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(wall_dir,
                                                 controller_name='SlideFurther',
                                                 goal_is_relative='1',
                                                 name="SlideFurther",
                                                 reference_frame="world",
                                                 v_max=slide_velocity))

        # Switch when goal is reached to trigger the relative slide further motion
        control_sequence.append(ha.JointConfigurationSwitch('SlideToWall', 'SlideFurther', controller='SlideToWall',
                                                            epsilon=str(math.radians(7.))))

        # Force/Torque switch for the additional relative slide further
        control_sequence.append(ha.ForceTorqueSwitch('SlideFurther',
                                                     mode_name_hand_closing,
                                                     name='ForceSwitch',
                                                     goal=wall_force_threshold,
                                                     norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion="THRESH_UPPER_BOUND",
                                                     goal_is_relative='1',
                                                     frame_id='world',
                                                     frame=corner_frame_alpha_zero,
                                                     port='2'))
    else:
        # Sliding motion (entirely world space controlled)
        control_sequence.append(
            ha.InterpolatedHTransformControlMode(wall_dir, controller_name='SlideToWall', goal_is_relative='1',
                                                 name="SlideToWall", reference_frame="world",
                                                 v_max=slide_velocity))

    # 7b. Switch when the f/t sensor is triggered with normal force from wall
    #     (in both cases joint trajectory or op-space control)
    # TODO arne: needs tuning
    # EDITED
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', mode_name_hand_closing, 'ForceSwitch',
                                                 goal=wall_force_threshold,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=corner_frame_alpha_zero, port='2'))
    # /EDITED

    # 8. Maintain contact while closing the hand
    # apply force on object while closing the hand
    # TODO arne: validate these values
    desired_displacement = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    force_gradient = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.005], [0.0, 0.0, 0.0, 1.0]])
    desired_force_dimension = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    control_sequence.append(ha.HandControlMode_ForceHT(name=mode_name_hand_closing, synergy=hand_synergy,
                                                       desired_displacement=desired_displacement,
                                                       force_gradient=force_gradient,
                                                       desired_force_dimension=desired_force_dimension))

    # 8b. Switch when hand closing duration ends
    control_sequence.append(ha.TimeSwitch(mode_name_hand_closing, 'PostGraspRotate', duration=hand_closing_duration))

    # 9. Move hand after closing and before lifting it up
    # relative to current hand pose
    control_sequence.append(
        ha.HTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate',
                                 goal_is_relative='1', ))

    # 9b. Switch when hand reaches post grasp pose
    control_sequence.append(ha.FramePoseSwitch('PostGraspRotate', 'GoUp_1', controller='PostGraspRotate',
                                               epsilon='0.01', goal_is_relative='1', reference_frame='EE'))

    return control_sequence  # TODO what about rviz_frames?

