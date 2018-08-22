import tf_conversions.posemath as pm
#from xper_data import srv as xper_srv
from tf import transformations as tra
import numpy as np
import math
import rospy
import hatools.components as ha

def getParam(obj_type_params, obj_params, paramKey):
    param = obj_type_params.get(paramKey)
    if param is None:
        param = obj_params.get(paramKey)
    if param is None:
         raise Exception("Param: " + paramKey + " does not exist for this object and there is no generic value defined")
    return param

def create_surface_grasp(object_frame, bounding_box, handarm_params, object_type, ifco_in_base, pre_grasp_pose = None):

    # Get the parameters from the handarm_parameters.py file
    obj_type_params = {}
    obj_params = {}
    if (object_type in handarm_params['surface_grasp']):            
        obj_type_params = handarm_params['surface_grasp'][object_type]
    if 'object' in handarm_params['surface_grasp']:
        obj_params = handarm_params['surface_grasp']['object']

    hand_transform = getParam(obj_type_params, obj_params, 'hand_transform')
    downward_force = getParam(obj_type_params, obj_params, 'downward_force')
    ee_in_goal_frame = getParam(obj_type_params, obj_params, 'ee_in_goal_frame')

    down_IFCO_speed = handarm_params['down_IFCO_speed']
    
    zflip_transform = tra.rotation_matrix(math.radians(180.0), [0, 0, 1])
    if object_frame[0][1]<0:
        object_frame = object_frame.dot(zflip_transform)

    if pre_grasp_pose is None:
        # Set the initial pose above the object
        goal_ = np.copy(object_frame)
        goal_ = goal_.dot(hand_transform) #this is the pre-grasp transform of the signature frame expressed in the world
        goal_ = goal_.dot(ee_in_goal_frame)
    else:
        goal_ = pre_grasp_pose

    # call_xper = rospy.ServiceProxy('pregrasp_pose', xper_srv.ProvidePreGraspPose)
    # res = call_xper(pm.toMsg(pm.fromMatrix(ifco_in_base)), pm.toMsg(pm.fromMatrix(object_frame)), pm.toMsg(pm.fromMatrix(goal_)), "surface")
    # print("REACHABILITY & EXPERIMENTS node proposes: ")
    # print("approach_direction: " + str(res.approach_direction))
    # print("hand_orientation: " + str(res.hand_orientation))
    # print("plane_orientation: " + str(res.plane_orientation))
    # print(pm.toMatrix(pm.fromMsg(res.reachable_hand_pose)))


    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the EE frame
    down_IFCO_twist = tra.translation_matrix([0, 0, down_IFCO_speed]);

    # Set the frames to visualize with RViz
    rviz_frames = []
    rviz_frames.append(object_frame)
    rviz_frames.append(goal_)
    # rviz_frames.append(pm.toMatrix(pm.fromMsg(res.reachable_hand_pose)))

    # assemble controller sequence
    control_sequence = []

    # 1. Go above the object - Pregrasp
    control_sequence.append(ha.InterpolatedHTransformControlMode(goal_, controller_name = 'GoAboveObject', goal_is_relative='0', name = 'Pregrasp'))
 
    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('Pregrasp', 'StayStill', controller = 'GoAboveObject', epsilon = '0.01'))
 
    # 2. Go to gravity compensation 
    control_sequence.append(ha.GravityCompensationMode(name = 'StayStill'))

    # 2b. Wait for a bit to allow vibrations to attenuate
    control_sequence.append(ha.TimeSwitch('StayStill', 'GoDown', duration = handarm_params['stay_still_duration']))
 
    # 3. Go down onto the object (relative in EE frame) - Godown
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(down_IFCO_twist,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="EE",
                                             v_max=down_IFCO_speed))

    # 3b. Switch when force-torque sensor is triggered
    force  = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'softhand_close',
                                                 goal = force,
                                                 norm_weights = np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion = "THRESH_UPPER_BOUND",
                                                 goal_is_relative = '1',
                                                 frame_id = 'world',
                                                 port = '2'))

    if handarm_params['SimplePositionControl']:
        # if hand is controlled in position mode, then call general hand controller
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = 'softhand_close', synergy = '1'))
        # 4b. Switch when hand closing time ends
        control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration = handarm_params['hand_closing_duration']))
    
    elif handarm_params['ImpedanceControl']:
        # if hand is controlled in impedance mode, then call IIT's controller
        kp = getParam(obj_type_params, obj_params, 'kp')
        control_sequence.append(ha.ros_PisaIIThandControlMode(goal = np.array([1.0]), kp=np.array([kp]), hand_max_aperture = handarm_params['hand_max_aperture'], name  = 'softhand_close', 
            bounding_box=np.array([bounding_box.x, bounding_box.y, bounding_box.z]), object_weight=np.array([0.4]), object_type='object', object_pose=object_frame))
        # 4b. Switch when hand closing time ends
        control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration = handarm_params['hand_closing_duration']))
    elif handarm_params['IMUGrasp']:
        # if hand is controlled with PISA's controller, do some extra steps (or not)
        # call PISA controller
        control_sequence.append(ha.IMUGraspControlMode(object_frame, name = 'softhand_close'))
        # 4b. Switch when hand closing time ends
        control_sequence.append(ha.TimeSwitch('softhand_close', 'GoUp', duration = handarm_params['compensation_duration']))

    return control_sequence, rviz_frames


# ================================================================================================
def create_wall_grasp(object_frame, bounding_box, wall_frame, handarm_params, object_type, ifco_in_base, pre_grasp_pose = None):

    # Get the parameters from the handarm_parameters.py file
    obj_type_params = {}
    obj_params = {}
    if object_type in handarm_params['wall_grasp']:            
        obj_type_params = handarm_params['wall_grasp'][object_type]
    if 'object' in handarm_params['wall_grasp']:
        obj_params = handarm_params['wall_grasp']['object']

    hand_transform = getParam(obj_type_params, obj_params, 'hand_transform')
    downward_force = getParam(obj_type_params, obj_params, 'downward_force')
    wall_force = getParam(obj_type_params, obj_params, 'wall_force')
    slide_IFCO_speed = getParam(obj_type_params, obj_params, 'slide_speed')
    pre_approach_transform = getParam(obj_type_params, obj_params, 'pre_approach_transform')

    vision_params = {}
    if object_type in handarm_params:
        vision_params = handarm_params[object_type]
    offset = getParam(vision_params, handarm_params['object'], 'obj_bbox_uncertainty_offset')
    if abs(object_frame[:3,0].dot(wall_frame[:3,0])) > abs(object_frame[:3,1].dot(wall_frame[:3,0])):
        pre_approach_transform[2,3] = pre_approach_transform[2,3] - bounding_box.y/2 - offset 
    else:
        pre_approach_transform[2,3] = pre_approach_transform[2,3] - bounding_box.x/2 - offset

    post_grasp_transform = getParam(obj_type_params, obj_params, 'post_grasp_transform')

    rotate_time = handarm_params['rotate_duration']
    down_IFCO_speed = handarm_params['down_IFCO_speed']

    # Set the twists to use TRIK controller with

    # Down speed is positive because it is defined on the world frame
    down_IFCO_twist = tra.translation_matrix([0, 0, -down_IFCO_speed]);
    # Slide speed is positive because it is defined on the EE frame
    slide_IFCO_twist = tra.translation_matrix([0, 0, slide_IFCO_speed]);

    
    rviz_frames = []

    if pre_grasp_pose is None:
        # this is the EC frame. It is positioned like object and oriented to the wall
        ec_frame = np.copy(wall_frame)
        ec_frame[:3, 3] = tra.translation_from_matrix(object_frame)
        ec_frame = ec_frame.dot(hand_transform)

        pre_approach_pose = ec_frame.dot(pre_approach_transform)
    else:
        pre_approach_pose = pre_grasp_pose

    # call_xper = rospy.ServiceProxy('pregrasp_pose', xper_srv.ProvidePreGraspPose)
    # res = call_xper(pm.toMsg(pm.fromMatrix(ifco_in_base)), pm.toMsg(pm.fromMatrix(object_frame)), pm.toMsg(pm.fromMatrix(pre_approach_pose)), "wall")
    # print("REACHABILITY & EXPERIMENTS node proposes: ")
    # print("approach_direction: " + str(res.approach_direction))
    # print("hand_orientation: " + str(res.hand_orientation))
    # print("plane_orientation: " + str(res.plane_orientation))
    # print(pm.toMatrix(pm.fromMsg(res.reachable_hand_pose)))

    # Rviz debug frames
    rviz_frames.append(object_frame)
    rviz_frames.append(pre_approach_pose)
    # rviz_frames.append(pm.toMatrix(pm.fromMsg(res.reachable_hand_pose)))


    control_sequence = []

    # 1. Go above the object
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(pre_approach_pose, controller_name='GoAboveObject', goal_is_relative='0',
                                             name="PreGrasp"))

    # 1b. Switch when hand reaches the goal pose
    control_sequence.append(ha.FramePoseSwitch('PreGrasp', 'GoDown', controller='GoAboveObject', epsilon='0.01'))

    # 2. Go down onto the object/table, in world frame
    control_sequence.append( ha.InterpolatedHTransformControlMode(down_IFCO_twist,
                                             controller_name='GoDown',
                                             goal_is_relative='1',
                                             name="GoDown",
                                             reference_frame="world"))

    # 2b. Switch when force threshold is exceeded
    force = np.array([0, 0, downward_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('GoDown',
                                                 'SlideToWall',
                                                 goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND",
                                                 goal_is_relative='1',
                                                 frame_id='world',
                                                 port='2'))

    # 3. Go towards the wall to slide object to wall
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(slide_IFCO_twist, controller_name='SlideToWall', goal_is_relative='1',
                                             name="SlideToWall", reference_frame="EE"))

    # 3b. Switch when the f/t sensor is triggered with normal force from wall
    force = np.array([0, 0, wall_force, 0, 0, 0])
    control_sequence.append(ha.ForceTorqueSwitch('SlideToWall', 'softhand_close', 'ForceSwitch', goal=force,
                                                 norm_weights=np.array([0, 0, 1, 0, 0, 0]),
                                                 jump_criterion="THRESH_UPPER_BOUND", goal_is_relative='1',
                                                 frame_id='world', frame=wall_frame, port='2'))

    # 4. Close the hand
    
    if handarm_params['SimplePositionControl']:
        # if hand is controlled in position mode, then call general hand controller
        control_sequence.append(ha.GeneralHandControlMode(goal = np.array([1]), name  = 'softhand_close', synergy = '1'))
        # 4b. Switch when hand closing time ends
        control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration = handarm_params['hand_closing_duration']))
    
    elif handarm_params['ImpedanceControl']:
        # if hand is controlled in impedance mode, then call IIT's controller
        kp = getParam(obj_type_params, obj_params, 'kp')
        control_sequence.append(ha.ros_PisaIIThandControlMode(goal = np.array([1.0]), kp=np.array([kp]), hand_max_aperture = handarm_params['hand_max_aperture'], name  = 'softhand_close', 
            bounding_box=np.array([bounding_box.x, bounding_box.y, bounding_box.z]), object_weight=np.array([0.4]), object_type='object', object_pose=object_frame))
        # 4b. Switch when hand closing time ends
        control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration = handarm_params['hand_closing_duration']))
    elif handarm_params['IMUGrasp']:
        # if hand is controlled with PISA's controller, do some extra steps (or not)
        # call PISA controller
        control_sequence.append(ha.IMUGraspControlMode(object_frame, name = 'softhand_close'))
        # 4b. Switch when hand closing time ends
        control_sequence.append(ha.TimeSwitch('softhand_close', 'PostGraspRotate', duration = handarm_params['compensation_duration']))

    # 5. Rotate hand after closing and before lifting it up relative to current hand pose
    control_sequence.append(
        ha.InterpolatedHTransformControlMode(post_grasp_transform, controller_name='PostGraspRotate', name='PostGraspRotate', goal_is_relative='1', reference_frame='EE'))

    # 5b. Switch when hand rotated
    control_sequence.append(ha.TimeSwitch('PostGraspRotate', 'GoUp', duration = rotate_time))

    return control_sequence, rviz_frames