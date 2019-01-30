import numpy as np
import math
import hatools.components as ha

def get_placement_recipe(handarm_params, handarm_type):

    place_time = handarm_params['place_duration']
    place_speed = handarm_params['place_speed']
    view_joint_config = handarm_params['view_joint_config']


    # The twists are defined on the world frame
    down_twist = np.array([0, 0, -place_speed, 0, 0, 0]); 
    up_twist =  np.array([0, 0, place_speed, 0, 0, 0]);

    # assemble controller sequence
    control_sequence = []

    # 1. Place in tote
    control_sequence.append(ha.CartesianVelocityControlMode(down_twist, controller_name = 'PlaceInTote', name = 'PlaceInTote', reference_frame="world"))
 
    # 1b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('PlaceInTote', 'softhand_open', duration = place_time))

    # 2. Release SKU
    if "ClashHand" in handarm_type:
        # Load the proper params from handarm_parameters.py
        # Open hand goal
        goal_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        control_sequence.append(ha.ros_CLASHhandControlMode(goal=goal_open, behaviour='GotoPos',  name='softhand_open'))
        
        # 2b. Switch when hand opening time ends
        control_sequence.append(ha.TimeSwitch('softhand_open', 'unstiffen_hand_again', duration = 0.5))

        # 2.1. Trigger pre-shaping the hand and/or pretension
        control_sequence.append(ha.ros_CLASHhandControlMode(name='unstiffen_hand_again', behaviour='SetPretension'))
    
        # 2.2. Time to trigger pre-shape
        control_sequence.append(ha.TimeSwitch('unstiffen_hand_again', 'get_out_of_tote', duration = 1))


    else:
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = 1))

        # 2b. Switch when hand opening time ends
        control_sequence.append(ha.TimeSwitch('softhand_open', 'get_out_of_tote', duration = 0.5))

    # 3. Get out of the tote
    control_sequence.append(ha.CartesianVelocityControlMode(up_twist, controller_name = 'GetOutOfTote', name = 'get_out_of_tote', reference_frame="world"))
 
    # 3b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('get_out_of_tote', 'go_to_view_config', duration = place_time))

    # 4. View config above ifco
    control_sequence.append(ha.PlanningJointControlMode(view_joint_config, name='go_to_view_config', controller_name='viewJointCtrl'))

    # 4b. Joint config switch
    control_sequence.append(ha.JointConfigurationSwitch('go_to_view_config', 'finished', controller='viewJointCtrl', epsilon=str(math.radians(7.))))

    # 4c. Switch if no plan was found
    control_sequence.append(ha.RosTopicSwitch('go_to_view_config', 'finished', ros_topic_name='controller_state', ros_topic_type='UInt8', goal=np.array([1.])))

    # 5. Block joints to finish motion
    control_sequence.append(ha.BlockJointControlMode(name  = 'finished'))

    return control_sequence