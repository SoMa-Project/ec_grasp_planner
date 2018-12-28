import numpy as np
import hatools.components as ha

def get_placement_recipe(chosen_object, handarm_params, grasp_type):

    object_type = chosen_object['type']
    # Get the relevant parameters for hand object combination
    if object_type in handarm_params[grasp_type]:
        params = handarm_params[grasp_type][object_type]
    else:
        params = handarm_params[grasp_type]['object']

    place_time = params['place_duration']
    down_speed = params['down_speed']

    # The twists are defined on the world frame
    down_twist = np.array([0, 0, -down_speed, 0, 0, 0]);  
 
    # assemble controller sequence
    control_sequence = []

    # 1. Place in IFCO
    control_sequence.append(ha.CartesianVelocityControlMode(down_twist, controller_name = 'PlaceInIFCO', name = 'PlaceInIFCO', reference_frame="world"))
 
    # 1b. Switch after a certain amount of time
    control_sequence.append(ha.TimeSwitch('PlaceInIFCO', 'softhand_open', duration = place_time))

    # 2. Release SKU
    control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open', synergy = 1))

    # 2b. Switch when hand opening time ends
    control_sequence.append(ha.TimeSwitch('softhand_open', 'finished', duration = 0.5))

    # 3. Block joints to finish motion
    control_sequence.append(ha.BlockJointControlMode(name  = 'finished'))

    return control_sequence