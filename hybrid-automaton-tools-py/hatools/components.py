import numpy as np

from utils import dict_to_string

class HybridAutomaton(object):
    def __init__(self, name = 'MyHA', current_control_mode = 'MyCM'):
        self.properties = {'name': name, 'current_control_mode': current_control_mode}
        self.modes_and_switches = []
    
    def add(self, other):
        self.modes_and_switches.extend(other) if type(other) == list else self.modes_and_switches.append(other)
        return self
    
    def xml(self):
        return '<HybridAutomaton %s>%s</HybridAutomaton>' % (dict_to_string(self.properties), ''.join([i.xml() for i in self.modes_and_switches]))

class ControlMode(object):
    def __init__(self, name):
        self.properties = {'name': name}
        self.controlset = ControlSet()
    
    def set(self, controlset):
        self.controlset = controlset
        return self
    
    def xml(self):
        return '<ControlMode %s>%s</ControlMode>' % (dict_to_string(self.properties), self.controlset.xml())

class ControlSet(object):
    """Base class for a description of a control set.

    A control set can contain multiple controllers. It defines which type inverse dynamics and inverse kinematics to use.

    Args:
        type (str): What kind of control set this is.

    Attributes:
        properties (dict): A dictionary of control set specific properties.
    """
    def __init__(self, type = 'rxControlSet', name = 'default'):
        self.properties = {'name': name, 'type': type}
        self.controllers = []
    
    def add(self, other):
        self.controllers.extend(other) if type(other) == list else self.controllers.append(other)
        return self
    
    def xml(self):
        return '<ControlSet %s>%s</ControlSet>' % (dict_to_string(self.properties), ''.join([i.xml() for i in self.controllers]))

class Controller(object):
    """Base class for controller descriptions of a hybrid automaton.

    Note:
        All arguments are based on the members defined in hybrid_automaton/Controller.h.

    Args:
        type (str): What kind of controller this is.
        goal (numpy.array): The desired goal(s) of the controller.
        goal_is_relative (bool, optional): Whether the desired goal is specified in an absolute or relative frame of reference.
        name (str, optional): An identification of the controller.
        kp (numpy.array, optional): Gain parameters.
        kv (numpy.array, optional): Gain parameters.
        priority (double, optional): A priority which influences the controller's behavior in certain control sets.
        completion_times (np.array, optional): The desired time to arrive at the goal.
        v_max (np.array, optional): The maximum velocity to go to the goal.
        a_max (np.array, optional): The maximum acceleration to go to the goal.
        reinterpolation (bool, optional): Whether new desired goals are attached at to the end of the current queue of goals or whether they should replace them.

    Attributes:
        properties (dict): A dictionary of controller specific properties.
    """
    def __init__(self, type, goal, goal_is_relative = '1', name = 'goto', kp = np.array([300, 200, 150, 20, 10, 10, 10]),
                 kv = np.array([2, 4, 2, 0.8, 0.2, 0.2, 0.02]), priority = '0', completion_times = np.array([]),
                 v_max = np.array([]), a_max = np.array([]), reinterpolation = '0'):
        self.properties = {'type': type, 'goal': goal, 'goal_is_relative': goal_is_relative, 'name': name,
                           'kp': kp, 'kv': kv, 'priority': priority, 'completion_times': completion_times,
                           'v_max': v_max, 'a_max': a_max, 'reinterpolation': reinterpolation}
    
    def xml(self):
        return '<Controller %s/>' % (dict_to_string(self.properties))

class ControlSwitch(object):
    """Base class for control switch descriptions of a hybrid automaton.
    
    A control switch is represented by a directed edge connecting nodes that in turn represent control modes.
    
    Args:
        source (str): Control mode that is connected by this switch.
        target (str): Control mode that is connected by this switch.
        name (str, optional): Identification of the switch.
    """
    def __init__(self, source, target, name = ''):
        self.properties = {'name': name, 'source': source, 'target': target}
        self.conditions = []
    
    def add(self, other):
        self.conditions.extend(other) if type(other) == list else self.conditions.append(other)
        return self
    
    def xml(self):
        return '<ControlSwitch %s>%s</ControlSwitch>' % (dict_to_string(self.properties), ''.join([i.xml() for i in self.conditions]))

class JumpCondition(object):
    """Base class for jump condition descriptions of a hybrid automaton.
    
    A jump condition is a boolean statement of one of the following criteria:
    NORM_L1, NORM_L2, NORM_L_INF, NORM_ROTATION, NORM_TRANSFORM, THRESH_UPPER_BOUND, THRESH_LOWER_BOUND
    It compares some sensor value with some fixed value.
    
    Args:
        sensor_type (str): .
        goal (numpy.array): .
        controller (str, optional):
        jump_criterion (str, optional): {NORM_L1, NORM_L2, NORM_L_INF, NORM_ROTATION, NORM_TRANSFORM, THRESH_UPPER_BOUND, THRESH_LOWER_BOUND}
        norm_weights (numpy.array, optional): 
        epsilon (double, optional):
        negate (bool, optional): compare "<" vs ">="
        goal_is_relative (bool, optional):
        frame_ID (str): reference frame, only used for ForceTorqueSensor
        port (int): the port of the FT driver measurement
        frame (numpy.array): transformation that can be applied to the FT measurement
    """
    def __init__(self, sensor_type, goal = np.array([]), controller = '', jump_criterion = 'THRESH_LOWER_BOUND', negate = '0', norm_weights = np.array([]), epsilon = '0.8', goal_is_relative = '0', frame_id = '', port = '', reference_frame = '', frame = '' ):
        self.properties = {'controller': controller, 'epsilon': epsilon,
                           'goal_is_relative': goal_is_relative, 'jump_criterion': jump_criterion,
                           'negate': negate, 'norm_weights': norm_weights }
        if len(goal) > 0:  # only add it if specified, otherwise controller is used
            self.properties.update({'goal': goal})
        self.sensor_properties = {'type': sensor_type, 'frame_id': frame_id, 'port': port, 'reference_frame': reference_frame, 'frame': frame}
    
    def xml(self):
        return '<JumpCondition %s><Sensor %s/></JumpCondition>' % (dict_to_string(self.properties), dict_to_string(self.sensor_properties))

class GravityCompensationMode(ControlMode):
    def __init__(self, name = 'Floating'):
        super(GravityCompensationMode, self).__init__(name = name)
        self.controlset = ControlSet(type = 'rxControlSet')

class JointControlMode(ControlMode):
    def __init__(self, goal, goal_is_relative = '0', completion_times = np.array([]), name = 'GoTo', controller_name = 'GoToJointConfig'):
        super(JointControlMode, self).__init__(name = name)
        self.controlset = ControlSet(type = 'rxControlSet').add(JointController(goal, name = controller_name, goal_is_relative = goal_is_relative, completion_times = completion_times))

class HTransformControlMode(ControlMode):
    def __init__(self, goal, goal_is_relative = '1', completion_times = np.array([]), name = 'GoTo', controller_name = 'GoToCartesianConfig', joint_weights = None, null_space_posture = False):
        super(HTransformControlMode, self).__init__(name = name)
        if joint_weights is None and not null_space_posture:
            self.controlset = NakamuraControlSet().add(HTransformController(goal, goal_is_relative = goal_is_relative, name = controller_name, completion_times = completion_times))
        else:
            self.controlset = TPNakamuraControlSet(joint_weights = joint_weights).add(HTransformController(goal, goal_is_relative = goal_is_relative, name = controller_name, completion_times = completion_times))
        if null_space_posture:
            self.controlset.add(SubjointController(priority = 1, index = np.array(range(7))), goal_is_relative = '1', goal = np.zeros(7))

class InterpolatedHTransformControlMode(ControlMode):
    def __init__(self, goal, goal_is_relative = '1', completion_times = np.array([]),
                 name = 'GoTo', controller_name = 'GoToCartesianConfig',
                 joint_weights = None, null_space_posture = False, reference_frame = "EE", v_max = np.array([0.125,0.08])):
        super(InterpolatedHTransformControlMode, self).__init__(name = name)
        if joint_weights is None and not null_space_posture:
            self.controlset = NakamuraControlSet().add(InterpolatedHTransformController(goal, goal_is_relative = goal_is_relative,
                                                                                        name = controller_name,
                                                                                        completion_times = completion_times,
                                                                                        reference_frame = reference_frame,
                                                                                        v_max = v_max))
        else:
            self.controlset = TPNakamuraControlSet(joint_weights = joint_weights).add(InterpolatedHTransformController(goal, goal_is_relative = goal_is_relative,
                                                                                                                       name = controller_name,
                                                                                                                       completion_times = completion_times,
                                                                                                                       reference_frame = reference_frame,
                                                                                                                       v_max = v_max))
        if null_space_posture:
            self.controlset.add(SubjointController(priority = 1, index = np.array(range(7))), goal_is_relative = '1', goal = np.zeros(7))


class InterpolatedHTransformImpedanceControlMode(ControlMode):
    def __init__(self, goal, goal_is_relative = '1', completion_times = np.array([]), name = 'GoTo', controller_name = 'GoToCartesianConfig', joint_weights = None, damping = np.array([100,30,100,10,10,10]), mass = np.array([60,15,60,1,1,1]), stiffness = np.array([60,15,60,0.5,0.5,0.5]), force_relative_to_initial = '1' , js_kp = np.array([30, 20, 15, 20, 10, 10, 10]), js_kd = np.array([1, 2, 1, 0.4, 0.1, 0.1, 0.01]), null_space_posture = False, ):
        super(InterpolatedHTransformImpedanceControlMode, self).__init__(name = name)
        if joint_weights is None and not null_space_posture:
            self.controlset = NakamuraControlSet(js_kp = js_kp, js_kd = js_kd).add(InterpolatedHTransformImpedanceController(goal, goal_is_relative = goal_is_relative, name = controller_name, completion_times = completion_times, damping = damping, mass = mass, stiffness = stiffness, force_relative_to_initial = force_relative_to_initial ))
        else:
            self.controlset = TPNakamuraControlSet(joint_weights = joint_weights, js_kp = js_kp, js_kd = js_kd).add(InterpolatedHTransformImpedanceController(goal, goal_is_relative = goal_is_relative, name = controller_name, completion_times = completion_times, damping = damping, mass = mass, stiffness = stiffness, force_relative_to_initial = force_relative_to_initial ))
        if null_space_posture:
            self.controlset.add(SubjointController(priority = 1, index = np.array(range(7))), goal_is_relative = '1', goal = np.zeros(7))

class ForceHTransformControlMode(ControlMode):
    def __init__(self, desired_displacement, force_gradient, desired_force_dimension = np.array([0, 0, 1, 0, 0, 0]), name = 'Slide'):
        super(ForceHTransformControlMode, self).__init__(name = name)
        self.controlset = NakamuraControlSet().add(ForceHTransformController(desired_displacement = desired_displacement, force_gradient = force_gradient, desired_force_dimension = desired_force_dimension))

class HandControlMode(GravityCompensationMode):
    def __init__(self, name = 'MoveFingers', goal = 0, controller_name = 'MoveFingers'):
        super(HandControlMode,self).__init__(name = name)
        self.controlset.add(HandController(goal = goal, name = controller_name))

# TODO:: drop this when planner.py has the nice implementation!
class HandControlMode_ForceHT(ForceHTransformControlMode):
    def __init__(self, desired_displacement, force_gradient, desired_force_dimension = np.array([0, 0, 1, 0, 0, 0]), name = 'grasp', controller_name = 'MoveFingers', synergy = 0,):
        super(HandControlMode_ForceHT, self).__init__(name = name, desired_displacement = desired_displacement, force_gradient = force_gradient, desired_force_dimension = desired_force_dimension)
        self.controlset.add(RBOHandController(goal =  np.array([[synergy, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0]])))               
        
class RBOHandControlMode(GravityCompensationMode):
    def __init__(self, goal = np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0]]), valve_times = np.array([[0, 5], [0, 5], [0, 5], [0, 5], [0, 5], [0, 5]]), goal_is_relative = '1', name = 'MoveFingers'):
        super(RBOHandControlMode, self).__init__(name = name)
        self.controlset.add(RBOHandController(goal = goal, valve_times = valve_times, goal_is_relative = goal_is_relative))

class SimpleRBOHandControlMode(ControlMode):
    def __init__(self, goal = np.array(1), name = 'MoveFingers'):
        super(SimpleRBOHandControlMode, self).__init__(name = name)
        self.controlset.add(SimpleRBOHandController(goal = goal))

class NakamuraControlSet(ControlSet):
    #" name="" js_kd="[7,1]1;2;1;0.4;0.1;0.1;0.01" js_kp="[7,1]30;20;15;20;10;10;10
    def __init__(self, name = 'nakamura_set', js_kp = np.array([300, 200, 150, 20, 10, 10, 10]), js_kd = np.array([2, 4, 2, 0.8, 0.2, 0.2, 0.02])):
        super(NakamuraControlSet, self).__init__(name = name, type = 'NakamuraControlSet')
        self.properties.update({'js_kp': js_kp, 'js_kd': js_kd})

class TPNakamuraControlSet(ControlSet):
    #" name="" js_kd="[7,1]1;2;1;0.4;0.1;0.1;0.01" js_kp="[7,1]30;20;15;20;10;10;10
    def __init__(self, name = 'nakamura_set', js_kp = np.array([300, 200, 150, 20, 10, 10, 10]), js_kd = np.array([2, 4, 2, 0.8, 0.2, 0.2, 0.02]), joint_weights = None):
        super(TPNakamuraControlSet, self).__init__(name = name, type = 'TPNakamuraControlSet')
        self.properties.update({'js_kp': js_kp, 'js_kd': js_kd})
        if joint_weights is not None:
            self.properties.update({ 'joint_weights': joint_weights})

class JointController(Controller):
    def __init__(self, goal, name = 'GoToJointConfig', kp = np.array([300, 200, 150, 20, 10, 10, 5]), kv = np.array([2, 4, 2, 0.8, 0.2, 0.2, 0.15]), goal_is_relative = '0', completion_times = np.array([]), interpolation_type = 'cubic', v_max = np.ones(7) * 0.2, a_max = np.array([]), reinterpolation = '1'):
        super(JointController, self).__init__('InterpolatedJointController', goal, name = name, completion_times = completion_times, v_max = v_max, a_max = a_max, reinterpolation = reinterpolation, kp = kp, kv = kv, goal_is_relative = goal_is_relative)
        self.properties['interpolation_type'] = interpolation_type

InterpolatedJointController = JointController

class SubjointController(Controller):
    def __init__(self, goal, index, name = 'GoToSubjointConfig', kp = np.array([300, 200, 150, 20, 10, 10, 5]), kv = np.array([2, 4, 2, 0.8, 0.2, 0.2, 0.15]), goal_is_relative = '0', completion_times = np.array([]), interpolation_type = 'cubic', v_max = np.ones(7) * 0.2, a_max = np.array([]), reinterpolation = '0'):
        super(SubjointController, self).__init__('InterpolatedSubjointController', goal, name = name, completion_times = completion_times, v_max = v_max, a_max = a_max, reinterpolation = reinterpolation, kp = kp, kv = kv, goal_is_relative = goal_is_relative)
        self.properties['interpolation_type'] = interpolation_type
        self.properties['index'] = index

class HTransformController(Controller):
    def __init__(self, goal, name = 'GoToCartesianConfig', operational_frame = 'EE', goal_is_relative = '1',
                 interpolation_type = 'cubic', v_max = np.array([0.125, 0.08]), a_max = np.array([]),
                 completion_times = np.array([]), reinterpolation = '0',
                 kp = np.array([0, 0, 0, 0, 0, 0]), kv = np.array([10, 10, 10, 10, 10, 10]), priority = '1'):
        super(HTransformController, self).__init__('InterpolatedHTransformController', goal, name = name,
                                                   completion_times = completion_times, v_max = v_max, a_max = a_max,
                                                   reinterpolation = reinterpolation, kp = kp, kv = kv,
                                                   goal_is_relative = goal_is_relative, priority = priority)
        self.properties.update({'interpolation_type': interpolation_type, 'operational_frame': operational_frame})

class InterpolatedHTransformController(Controller):
    def __init__(self, goal, name = 'GoToCartesianConfig', operational_frame = 'EE', goal_is_relative = '1',
                 interpolation_type = 'cubic', v_max = np.array([0.125, 0.08]), a_max = np.array([]),
                 completion_times = np.array([]), reinterpolation = '1', kp = np.array([0, 0, 0, 0, 0, 0]),
                 kv = np.array([10, 10, 10, 10, 10, 10]), priority = '1', reference_frame = "EE"):
        super(InterpolatedHTransformController, self).__init__('InterpolatedHTransformController',
                                                               goal, name = name, completion_times = completion_times,
                                                               v_max = v_max, a_max = a_max, reinterpolation = reinterpolation,
                                                               kp = kp, kv = kv, goal_is_relative = goal_is_relative, priority = priority)
        self.properties.update({'interpolation_type': interpolation_type, 'operational_frame': operational_frame, 'reference_frame': reference_frame})

class HTransformImpedanceController(Controller):
    def __init__(self, goal, name = 'GoToCartesianImpedanceConfig', operational_frame = 'EE', goal_is_relative = '1', kp = np.array([0, 0, 0, 0, 0, 0]), kv = np.array([10, 10, 10, 10, 10, 10]), damping = np.array([100,30,100,10,10,10]), mass = np.array([60,15,60,1,1,1]), stiffness = np.array([60,15,60,0.5,0.5,0.5]), force_relative_to_initial = '1' ):
        super(HTransformImpedanceController, self).__init__('HTransformImpedanceController', goal, name = name, kp = kp, kv = kv, goal_is_relative = goal_is_relative)
        self.properties.update({'operational_frame': operational_frame, 'damping': damping, 'mass': mass, 'stiffness': stiffness, 'force_relative_to_initial': force_relative_to_initial})

class InterpolatedHTransformImpedanceController(Controller):
    def __init__(self, goal, name = 'GoToCartesianImpedanceConfig', operational_frame = 'EE', goal_is_relative = '1', interpolation_type = 'cubic', v_max = np.array([0.1, 0.08]), a_max = np.array([]), completion_times = np.array([]), reinterpolation = '0', kp = np.array([0, 0, 0, 0, 0, 0]), kv = np.array([10, 10, 10, 10, 10, 10]), damping = np.array([100,30,100,10,10,10]), mass = np.array([60,15,60,1,1,1]), stiffness = np.array([60,15,60,0.5,0.5,0.5]), force_relative_to_initial = '1' ):
        super(InterpolatedHTransformImpedanceController, self).__init__('InterpolatedHTransformImpedanceController', goal, name = name, completion_times = completion_times, v_max = v_max, a_max = a_max, reinterpolation = reinterpolation, kp = kp, kv = kv, goal_is_relative = goal_is_relative)
        self.properties.update({'interpolation_type': interpolation_type, 'operational_frame': operational_frame, 'damping': damping, 'mass': mass, 'stiffness': stiffness, 'force_relative_to_initial': force_relative_to_initial})

class ForceHTransformController(Controller):
    def __init__(self, name = 'slide', desired_min_force = -0.5, desired_max_force = 0.5, desired_force_dimension=np.array([1, 0, 1, 0, 0, 0]), desired_displacement = np.eye(4), desired_distance = 0.15, force_gradient = np.eye(4), kp = np.array([0, 0, 0, 0, 0, 0, 0]), kv = np.array([10, 10, 10, 10, 10, 10]), interpolation_type = 'linear', goal_is_relative = '1'):
        super(ForceHTransformController, self).__init__('ForceHTransformController', np.eye(4), name = name, kp = kp, kv = kv, goal_is_relative = goal_is_relative)
        self.properties.update({'desired_min_force': desired_min_force, 'desired_max_force': desired_max_force, 'desired_force_dimension': desired_force_dimension, 'desired_displacement': desired_displacement, 'desired_distance': desired_distance, 'force_gradient': force_gradient, 'interpolation_type': interpolation_type, 'operational_frame': 'EE'})

class HandController(Controller):
    def __init__(self, name = 'move_fingers', goal = 0, goal_is_relative = 0):
        super(HandController, self).__init__('HandControlelr', goal, name = name, goal_is_relative = 0)

class RBOHandController(Controller):
    def __init__(self, name = 'move_fingers', goal = np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0]]), valve_times = np.array([[0, 5], [0, 5], [0, 5], [0, 5], [0, 5], [0, 5]]), goal_is_relative = '1'):
        super(RBOHandController, self).__init__('SoftHandController', goal, name = name, goal_is_relative = '1')
        self.properties.update({'valve_times': valve_times})

class SimpleRBOHandController(Controller):
    def __init__(self, name = 'simpleRBOHandController', goal = np.array([1])):
        super(SimpleRBOHandController, self).__init__('SimpleRBOHandController', goal,name = name)

class ForceTorqueSwitch(ControlSwitch):
    def __init__(self, source, target, name = 'ForceTorqueSwitch', goal = np.zeros(6), jump_criterion = 'THRESH_LOWER_BOUND', negate = '0', goal_is_relative = '1', frame_id = 'EE', norm_weights = np.array([]), epsilon = '0.0', frame=np.identity(4), port ='0'):
        super(ForceTorqueSwitch, self).__init__(source, target, name = name)
        self.conditions.append(JumpCondition('ForceTorqueSensor', goal = goal, jump_criterion = jump_criterion, negate = negate, goal_is_relative = goal_is_relative, norm_weights = norm_weights, epsilon = epsilon, frame_id = frame_id, port = port, frame = frame))

class JointConfigurationSwitch(ControlSwitch):
    def __init__(self, source, target, name = 'JointConfigurationSwitch', goal = np.array([]), norm_weights = np.ones(7), jump_criterion = 'NORM_L_INF', controller = '', goal_is_relative = '0', epsilon = '0.8'):
        super(JointConfigurationSwitch, self).__init__(source, target, name = name)
        self.conditions.append(JumpCondition('JointConfigurationSensor', goal = goal, controller = controller, jump_criterion = jump_criterion, norm_weights = norm_weights, goal_is_relative = goal_is_relative, epsilon = epsilon))

class FramePoseSwitch(ControlSwitch):
    def __init__(self, source, target, name = 'FramePoseSwitch', goal = np.array([]), norm_weights = np.array([0.1, 1.]),
                 jump_criterion = 'NORM_TRANSFORM', controller = '', goal_is_relative = '0', epsilon = '0.8', frame_id = 'EE', reference_frame = 'EE'):
        super(FramePoseSwitch, self).__init__(source, target, name = name)
        self.conditions.append(JumpCondition('FramePoseSensor', goal = goal, controller = controller,
                                             jump_criterion = jump_criterion, norm_weights = norm_weights,
                                             goal_is_relative = goal_is_relative,
                                             epsilon = epsilon, frame_id = frame_id, reference_frame = reference_frame))

class FrameDisplacementSwitch(ControlSwitch):
    def __init__(self, source, target, name = 'FrameDisplacementSwitch', epsilon = '0.1', negate = '1', goal = np.array([]), norm_weights = np.ones(3), jump_criterion = 'NORM_L2', controller = '', goal_is_relative = '1', frame_id = 'EE'):
        super(FrameDisplacementSwitch, self).__init__(source, target, name = name)
        self.conditions.append(JumpCondition('FrameDisplacementSensor', goal = goal, controller = controller, jump_criterion = jump_criterion, negate = negate, norm_weights = norm_weights, goal_is_relative = goal_is_relative, epsilon = epsilon, frame_id = frame_id))

class TimeSwitch(ControlSwitch):
    def __init__(self, source, target, name = 'TimeSwitch', epsilon = '0', duration = 1.0, jump_criterion = 'THRESH_UPPER_BOUND', goal_is_relative = '1'):
        super(TimeSwitch, self).__init__(source, target, name = name)
        self.conditions.append(JumpCondition('ClockSensor', goal = np.array([[duration]]), jump_criterion = jump_criterion, goal_is_relative = goal_is_relative, epsilon = epsilon))

class JointVelocitySwitch(ControlSwitch):
    def __init__(self, source, target, name = 'JointVelocitySwitch', goal = np.array([]), norm_weights = np.ones(7), jump_criterion = 'NORM_L1', controller = '', goal_is_relative = '0', epsilon = '0.8', negate = '0'):
        super(JointVelocitySwitch, self).__init__(source, target, name = name)
        self.conditions.append(JumpCondition('JointVelocitySensor', goal = goal, controller = controller, jump_criterion = jump_criterion, norm_weights = norm_weights, goal_is_relative = goal_is_relative, epsilon = epsilon, negate = negate))

class SubjointVelocitySwitch(ControlSwitch):
    def __init__(self, source, target, name = 'SubjointVelocitySwitch', goal = np.array([]), norm_weights = np.ones(7), jump_criterion = 'NORM_L1', controller = '', goal_is_relative = '0', epsilon = '0.8', negate = '0', index = np.ones(7)):
        super(SubjointVelocitySwitch, self).__init__(source, target, name = name)
        self.conditions.append(JumpCondition('SubjointVelocitySensor', goal = goal, controller = controller, jump_criterion = jump_criterion, norm_weights = norm_weights, goal_is_relative = goal_is_relative, epsilon = epsilon, negate = negate))
        self.conditions[0].sensor_properties.update({'index':index})
