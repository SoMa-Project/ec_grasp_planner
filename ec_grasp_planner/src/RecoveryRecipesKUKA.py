import numpy as np
import hatools.components as ha

def get_recovery_recipe(handarm_params, handarm_type, grasp_type, wall_frame = np.array([])):

    recovery_speed = handarm_params['recovery_speed']
    recovery_time = handarm_params['recovery_duration']
    downward_force = handarm_params['recovery_placement_force']

    up_world_twist = np.array([0, 0, recovery_speed, 0, 0, 0]) 
    down_world_twist = np.array([0, 0, -recovery_speed, 0, 0, 0])
    up_EE_twist = np.array([0, 0, -recovery_speed, 0, 0, 0])
    down_EE_twist = np.array([0, 0, recovery_speed, 0, 0, 0])

 
    # assemble controller sequence
    control_sequence = []

    if grasp_type == 'SurfaceGrasp':

        # Lift the hand if the CartesianVelocityControlMode crashed due to joint limits during approaching the object
        control_sequence.append(ha.CartesianVelocityControlMode(up_EE_twist, controller_name = 'GoUpHTransform', name = 'recovery_GoDownSG', reference_frame="EE"))

        # Switch to finished after some time
        control_sequence.append(ha.TimeSwitch('recovery_GoDownSG', 'finished', duration = recovery_time))    

        # If no plan to placement was found, start going down again to place the object back where it was grasped
        control_sequence.append(ha.CartesianVelocityControlMode(down_EE_twist, controller_name = 'GoUpHTransform', name = 'recovery_NoPlanFoundSurfaceGrasp', reference_frame="EE"))

        # force threshold that if reached will trigger the closing of the hand
        force = np.array([0, 0, downward_force, 0, 0, 0])
        
        # Switch when force-torque sensor is triggered
        control_sequence.append(ha.ForceTorqueSwitch('recovery_NoPlanFoundSurfaceGrasp',
                                                     'softhand_open_recovery_SurfaceGrasp',
                                                     goal = force,
                                                     norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion = "THRESH_UPPER_BOUND",
                                                     goal_is_relative = '1',
                                                     frame_id = 'world'))
    

        # Release SKU
        if "ClashHand" in handarm_type:
            # Load the proper params from handarm_parameters.py
            # Replace the BlockingJointControlMode with the CLASH hand control mode
            control_sequence.append(ha.BlockJointControlMode(name  = 'softhand_open_recovery_SurfaceGrasp'))
        else:
            control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open_recovery_SurfaceGrasp', synergy = 1))

        # Switch when hand opening time ends
        control_sequence.append(ha.TimeSwitch('softhand_open_recovery_SurfaceGrasp', 'recovery_GoDownSG', duration = 0.5))


    elif grasp_type == 'WallGrasp':
        # Lift the hand if the CartesianVelocityControlMode crashed due to joint limits during approaching the IFCO surface
        control_sequence.append(ha.CartesianVelocityControlMode(up_world_twist, controller_name = 'GoUpHTransform', name = 'recovery_GoDownWG', reference_frame="world"))

        # Switch to finished after some time
        control_sequence.append(ha.TimeSwitch('recovery_GoDownWG', 'finished', duration = recovery_time))    

        # Lift the hand if the CartesianVelocityControlMode crashed due to joint limits during approaching the IFCO wall
        control_sequence.append(ha.CartesianVelocityControlMode(up_EE_twist, controller_name = 'GoUpHTransform', name = 'recovery_SlideWG', reference_frame="EE"))

        # Switch to going up after a small amount of time
        control_sequence.append(ha.TimeSwitch('recovery_SlideWG', 'recovery_GoDownWG', duration = 1))    

        # If no plan to placement was found, start going down again to place the object back where it was grasped
        control_sequence.append(ha.CartesianVelocityControlMode(down_world_twist, controller_name = 'GoUpHTransform', name = 'recovery_NoPlanFoundWallGrasp', reference_frame="world"))

        # force threshold that if reached will trigger the closing of the hand
        force = np.array([0, 0, downward_force, 0, 0, 0])
        
        # Switch when force-torque sensor is triggered
        control_sequence.append(ha.ForceTorqueSwitch('recovery_NoPlanFoundWallGrasp',
                                                     'softhand_open_recovery_WallGrasp',
                                                     goal = force,
                                                     norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                     jump_criterion = "THRESH_UPPER_BOUND",
                                                     goal_is_relative = '1',
                                                     frame_id = 'world'))
    

        # Release SKU
        if "ClashHand" in handarm_type:
            # Load the proper params from handarm_parameters.py
            # Replace the BlockingJointControlMode with the CLASH hand control mode
            control_sequence.append(ha.BlockJointControlMode(name  = 'softhand_open_recovery_WallGrasp'))
        else:
            control_sequence.append(ha.GeneralHandControlMode(goal = np.array([0]), name  = 'softhand_open_recovery_WallGrasp', synergy = 1))

        # Switch when hand opening time ends
        control_sequence.append(ha.TimeSwitch('softhand_open_recovery_WallGrasp', 'slide_back_recovery', duration = 0.5))

        # Calculate the twist to slide back in the world frame
        # This is done because EE frame sliding back is no longer safe because of possible pre/post grasping rotations
        slide_back_linear_velocity = wall_frame[:3,:3].dot(np.array([0, 0, recovery_speed]))
        slide_back_world_twist = np.array([slide_back_linear_velocity[0], slide_back_linear_velocity[1], slide_back_linear_velocity[2], 0, 0, 0])

        # Slide back towards the IFCO center
        control_sequence.append(ha.CartesianVelocityControlMode(slide_back_world_twist, controller_name = 'GoUpHTransform', name = 'slide_back_recovery', reference_frame="world"))

        # Switch to going up after a small amount of time
        control_sequence.append(ha.TimeSwitch('slide_back_recovery', 'recovery_GoDownWG', duration = 1.5))    


    return control_sequence