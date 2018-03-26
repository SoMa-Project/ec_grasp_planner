import components as cp
import numpy as np

def gravity_compensation():
    return cp.HybridAutomaton(name = "GravityCompensation", current_control_mode = "Float").add(cp.GravityCompensationMode(name = "Float"))

def close_rbohand(valve_times = np.array([[0, 0.3], [0, 0.3], [0, 0.3], [0, 0.3], [0, 0.3], [0, 0.3]])):
    goal = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
    return cp.HybridAutomaton(name = "CloseRBOHand", current_control_mode = "Close").add(cp.ControlMode(name = "Close").set(cp.ControlSet(type = "rxControlSet").add(cp.RBOHandController(goal = goal, valve_times = valve_times))))

def open_rbohand(valve_times = np.array([[0, 5], [0, 5], [0, 5], [0, 5], [0, 5], [0, 5]])):
    goal = np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0]])
    return cp.HybridAutomaton(name = "OpenRBOHand", current_control_mode = "Open").add(cp.ControlMode(name = "Open").set(cp.ControlSet(type = "rxControlSet").add(cp.RBOHandController(goal = goal, valve_times = valve_times))))

def goto_joint(goal):
    cs = cp.ControlSet(type = "rxControlSet").add(cp.JointController(goal = goal))
    return cp.HybridAutomaton(name = "GotoJointConfiguration", current_control_mode = "Goto").add(cp.ControlMode(name = "Goto").set(cs))

def goto_cartesian(goal, **kwargs):
    cs = cp.NakamuraControlSet().add(cp.HTransformController(goal = goal, **kwargs))
    return cp.HybridAutomaton(name = "GotoCartesianPose", current_control_mode = "Goto").add(cp.ControlMode(name = "Goto").set(cs))

def single_control_mode(control_mode):
    return cp.HybridAutomaton(name = "SingleControlMode", current_control_mode = control_mode.properties['name']).add(control_mode)

def sequence_of_modes_and_switches(modes_and_switches):
    assert(len(modes_and_switches) > 1)
    for i, mode in enumerate(modes_and_switches[::2]):
        assert(isinstance(mode, cp.ControlMode))
        mode.properties['name'] = "Mode%i" % i
    for i, switch in enumerate(modes_and_switches[1::2]):
        assert(isinstance(switch, cp.ControlSwitch))
        switch.properties['source'] = "Mode%i" % i
        switch.properties['target'] = "Mode%i" % (i+1)
    
    return cp.HybridAutomaton(name = "Sequence", current_control_mode = modes_and_switches[0].properties['name']).add(modes_and_switches)

def sequence_of_modes_and_switches_with_safety_features(modes_and_switches):
    assert(len(modes_and_switches) > 1)
    # add for each mode all safety conditions
    for i, mode in enumerate(modes_and_switches[::2]):
        assert(isinstance(mode, cp.ControlMode))
        name = "safety_velocity_%i" % i
        modes_and_switches.append(cp.JointVelocitySwitch(mode.properties['name'], 'SafetyMode', name = name, goal = np.zeros(7),
                                                         norm_weights = np.array([1,1,1,1,1,1,0]), jump_criterion = '2',
                                                         epsilon = '3', negate = '1'))
        name = "safety_velocity_j7_%i" % i
        modes_and_switches.append(cp.JointVelocitySwitch(mode.properties['name'], 'SafetyMode', name = name, goal = np.zeros(7),
                                                         norm_weights = np.array([0,0,0,0,0,0,1]), jump_criterion = '2',
                                                         epsilon = '4', negate = '1'))
        name = "safety_force_%i" % i
        modes_and_switches.append(cp.ForceTorqueSwitch(mode.properties['name'], 'SafetyMode', name = name, goal = np.zeros(6),
                                                         norm_weights = np.array([1,1,1,0,0,0]), jump_criterion = '0',
                                                         epsilon = '35', negate = '1'))
        name = "safety_torque_%i" % i
        modes_and_switches.append(cp.ForceTorqueSwitch(mode.properties['name'], 'SafetyMode', name = name, goal = np.zeros(6),
                                                         norm_weights = np.array([0,0,0,1,1,1]), jump_criterion = '0',
                                                         epsilon = '2', negate = '1'))

        
    modes_and_switches.append(cp.GravityCompensationMode(name = 'SafetyMode'))
    
    return cp.HybridAutomaton(name = "Sequence", current_control_mode = modes_and_switches[0].properties['name']).add(modes_and_switches)
