import numpy as np
import math
import hatools.components as ha

def get_transport_recipe(handarm_params):

    lift_time = handarm_params['lift_duration']
    up_IFCO_speed = handarm_params['up_IFCO_speed']

    place_time = handarm_params['place_duration']
    down_tote_speed = handarm_params['down_tote_speed']

    # Up speed is also positive because it is defined on the world frame
    up_IFCO_twist = np.array([0, 0, up_IFCO_speed, 0, 0, 0]);

    # The twists are defined on the world frame
    down_tote_twist = np.array([0, 0, -down_tote_speed, 0, 0, 0]);  
    up_tote_twist = np.array([0, 0, down_tote_speed, 0, 0, 0]); 

    # Slide speed is negative because it is defined on the EE frame
    slide_IFCO_back_twist = np.array([0, 0, -handarm_params['recovery_slide_back_speed'], 0, 0, 0]);  

    # assemble controller sequence
    control_sequence = []

    # # Recovery from trik failing during guarded move towards a wall
    # control_sequence.append(ha.InterpolatedHTransformControlMode(slide_IFCO_back_twist, controller_name='RecoverSlide', goal_is_relative='1',
    #                                          name="RecoverSlide", reference_frame="EE"))
    # control_sequence.append(ha.TimeSwitch('RecoverSlide', 'RecoverDown', duration = handarm_params['recovery_slide_duration']))

    # # Recovery from trik failing during guarded move downwards

    # # 1. Lift upwards
    # control_sequence.append(ha.InterpolatedHTransformControlMode(up_IFCO_twist, controller_name = 'GoUpHTransform', name = 'RecoverDown', goal_is_relative='1', reference_frame="world"))
 
    # # 1b. Switch after a certain amount of time
    # control_sequence.append(ha.TimeSwitch('RecoverDown', 'softhand_open', duration = lift_time))

    # Normal transport

    # 1. Lift upwards
    control_sequence.append(ha.CartesianVelocityControlMode(up_IFCO_twist, controller_name = 'GoUpHTransform', name = 'GoUp', reference_frame="world"))
 
    # 1b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoUp', 'Preplacement1', duration = lift_time))

    # 2. Change the orientation to have the hand facing the Delivery tote
    control_sequence.append(ha.InterpolatedHTransformControlMode(handarm_params['pre_placement_pose'], controller_name = 'GoAbovePlacement', goal_is_relative='0', name = 'Preplacement1'))

    # 2b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Preplacement1', 'GoDown2', controller = 'GoAbovePlacement', epsilon = '0.1'))

    # # 3c. Recover if a plan is not found
    # control_sequence.append(ha.TimeSwitch('Preplacement3', 'RecoverSlide', duration = handarm_params['recovery_duration']))

    # 4. Go Down
    control_sequence.append(ha.CartesianVelocityControlMode(down_tote_twist, controller_name = 'GoToDropOff', name = 'GoDown2', reference_frame="world"))
 
    # 4b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('GoDown2', 'softhand_open', duration = place_time))

    # 5. Release SKU
    # if handarm_type == "ClashHandKUKA":
    #     speed = np.array([30]) 
    #     thumb_pos = np.array([0, -20, 0])
    #     diff_pos = np.array([-10, -10, 0])
    #     thumb_pretension = np.array([0]) 
    #     diff_pretension = np.array([0]) 
    #     mode = np.array([0])

    #     thumb_contact_force = np.array([0]) 
    #     thumb_grasp_force = np.array([0]) 
    #     diff_contact_force = np.array([0]) 
    #     diff_grasp_force = np.array([0])    
        
    #     force_feedback_ratio = np.array([0]) 
    #     prox_level = np.array([0]) 
    #     touch_level = np.array([0]) 
    #     command_count = np.array([2]) 

    #     control_sequence.append(ha.ros_CLASHhandControlMode(goal = np.concatenate((speed, thumb_pos, diff_pos, thumb_contact_force, 
    #                                                                             thumb_grasp_force, diff_contact_force, diff_grasp_force, 
    #                                                                             thumb_pretension, diff_pretension, force_feedback_ratio, 
    #                                                                             prox_level, touch_level, mode, command_count)), name  = 'softhand_open'))
        
    # else:
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = 1))

    # 5b. Switch when hand opening time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'initial', duration = handarm_params['hand_opening_duration']))

    # 6. Return to zero position
    control_sequence.append(ha.JointControlMode(goal = np.zeros(7), goal_is_relative = '0', name = 'initial', controller_name = 'GoToZeroController'))
    
    # 6b. Switch when zero position is reached
    control_sequence.append(ha.JointConfigurationSwitch('initial', 'finished', controller = 'GoToZeroController', epsilon = str(math.radians(7.0))))

    # 6. Block joints to finish motion
    control_sequence.append(ha.GravityCompensationMode(name  = 'finished'))

    return control_sequence