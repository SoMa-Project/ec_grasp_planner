#!/usr/bin/env python
import rospy
import roslib
import actionlib
import numpy as np
import subprocess
import os
import signal
import time
import sys
import argparse
import math
import yaml
import datetime

from random import randint
from random import uniform

import smach
import smach_ros

import tf
from tf import transformations as tra
import numpy as np

from geometry_msgs.msg import PoseStamped
from subprocess import call
from hybrid_automaton_msgs import srv
from hybrid_automaton_msgs.msg import HAMState

import hatools.components as ha
import hatools.cookbook as cookbook

class TouchFile(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['label_in', 'counter_in'])
    
    def execute(self, userdata):
        user_input = raw_input('Was the last trial a [s]uccess or [f]ailure? ')
        label = 'FAILURE'
        
        if user_input == 's':
            label = 'SUCCESS'
        
        call(['touch', userdata.label_in + '_' + str(userdata.counter_in) + '.' + label + '.label'])
        
        return 'done'
    
class LabelTrial(smach.State):
    def __init__(self, max_trials = -1):
        smach.State.__init__(self, outcomes=['next_trial','quit'],
                             input_keys=['label_in', 'counter_in'],
                             output_keys=['label_out', 'counter_out'])
        self.max_trials = max_trials

    def execute(self, userdata):
        print 'Please put "%s" into position %i.' % (userdata.label_in, userdata.counter_in + 1)
        new_label = raw_input('And type a new label if needed ("q" exits application): ')
        
        if new_label == 'q':
            return 'quit'
        elif new_label == '':
            userdata.label_out = userdata.label_in
            new_counter_value = userdata.counter_in + 1
        else:
            userdata.label_out = new_label
            new_counter_value = 0
        
        userdata.counter_out = new_counter_value
        
        if (self.max_trials > 0 and new_counter_value >= self.max_trials):
            return 'quit'
        
        return 'next_trial'

class StartBagRecording(smach.State):
    def __init__(self, topics, prefix = '', suffix = ''):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['label_in', 'counter_in', 'pid_in'],
                             output_keys=['pid_out'])
        self.topics = ' '.join(topics)
        self.prefix = prefix
        self.suffix = suffix

    def execute(self, userdata):
        #bag_filename = self.prefix + userdata.label_in + "_" + str(userdata.counter_in) + ".bag"
        bag_filename = self.prefix + userdata.label_in + self.suffix + ".bag"
        command = "rosbag record -O " + bag_filename + " " + self.topics
        p = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        userdata.pid_out = userdata.pid_in + [p.pid]
        return 'done'

class ExecuteMotion(smach.State):
    def __init__(self, motion_params):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['label_in', 'counter_in'])
        self.call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        self.motion_params = motion_params
    
    def execute(self, userdata):
        #jgoal = np.array([-2.06809, 1.0571,0.337885,2.15652,1.00802,0.462416,0.0475708])
        #go_above_table = ha.JointControlMode(jgoal, name = 'go_above_table', controller_name = 'GoToJointConfig', completion_times = np.array([[6.0]]))
        curree = np.array([[-0.19540635, -0.67037672,  0.71582918, -0.0451544 ],
                        [ 0.97680758, -0.06788595,  0.20307253, -0.489304  ],
                        [-0.08754036,  0.73890903,  0.66809441, -0.129727  ],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        #curree = np.dot(curree, tra.rotation_matrix(self.motion_params[userdata.counter_in], [1, 0, 0]))
        curree = np.dot(curree, tra.rotation_matrix(self.motion_params['angle_of_attack'], [1, 0, 0]))
        reposition_above_table = ha.HTransformControlMode(curree, name = 'reposition', goal_is_relative='0')
        
        goal = np.dot(tra.translation_matrix([0, 0, -0.3]), curree)
        go_down = ha.HTransformControlMode(goal, name = 'go_down', controller_name = 'GoToCartesianConfig', goal_is_relative='0')
        slide = ha.ControlMode(name = 'slide').set(ha.NakamuraControlSet().add(ha.ForceHTransformController(desired_distance = 0.25, desired_displacement=tra.translation_matrix([self.motion_params['sliding_speed'], 0, 0]), force_gradient=tra.translation_matrix([0, 0, 0.005]), desired_force_dimension=np.array([0, 0, 1, 0, 0, 0]))))
        
        ee_switch = ha.FramePoseSwitch('reposition', 'go_down', controller = 'GoToCartesianConfig', epsilon = '0.001')
        ft_switch = ha.ForceTorqueSwitch('go_down', 'slide', goal = np.array([0, 0, self.motion_params['downward_force'], 0, 0, 0]), norm_weights = np.array([0, 0, 1, 0, 0, 0]), jump_criterion = "THRESH_UPPER_BOUND", frame_id = 'odom', goal_is_relative = '1')
        distance_switch = ha.FrameDisplacementSwitch('slide', 'finished', epsilon = '0.15', negate = '1', goal = np.array([0, 0, 0]), goal_is_relative = '1', jump_criterion = "NORM_L2", frame_id = 'EE')
        wall_ft_switch = ha.ForceTorqueSwitch('slide', 'finished', goal = np.array([self.motion_params['wall_force'], 0, 0, 0, 0, 0]), norm_weights = np.array([1, 0, 0, 0, 0, 0]), jump_criterion = "THRESH_LOWER_BOUND", frame_id = 'odom', goal_is_relative = '1')
        #wall_ft_switch.add(distance_switch.conditions[0])
        finished = ha.GravityCompensationMode(name = 'finished')
        
        myha = ha.HybridAutomaton(current_control_mode='reposition').add([reposition_above_table, ee_switch, go_down, ft_switch, slide, wall_ft_switch, finished])
        
        self.call_ha(myha.xml())
        #userdata.counter_in
        return 'done'

class KillPG(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done'],
                             input_keys=['pid_in'],
                             output_keys=['pid_out'])

    def execute(self, userdata):
        if len(userdata.pid_in) > 0:
            print "killing! "
            for pid in userdata.pid_in:
                print pid
                os.killpg(pid, signal.SIGINT)
            userdata.pid_out = []
        else:
            print "Nothing to kill!"
        return 'done'

class ReturnToStart(smach.State):
    def __init__(self, motion_params):
        smach.State.__init__(self, outcomes=['done'])
        self.call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        
        jgoal = np.array([-2.06809, 1.0571,0.337885,2.15652,1.00802,0.462416,0.0475708])
        go_above_table = ha.JointControlMode(jgoal, name = 'go_above_table')
        go_above_table.controlset.add(ha.RBOHandController(goal = np.array([[-1,0,0],[-1,0,0],[-1,1,0],[-1,1,0],[-1,1,0],[-1,1,0]]), valve_times = np.array([[0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']], [0,5,5.+motion_params['finger_inflation']]]), goal_is_relative = '1'))
        joint_switch = ha.JointConfigurationSwitch('go_above_table', 'waiting', controller = 'GoToJointConfig', epsilon = str(math.radians(8.0)))
        joint_switch.add(ha.JumpCondition('ClockSensor', goal = np.array([[5.0 + 0.5]]), jump_criterion = 'THRESH_UPPER_BOUND', goal_is_relative = '1', epsilon = '0'))
        waiting = ha.GravityCompensationMode(name = 'waiting')
                
        self.return_ha = ha.HybridAutomaton(current_control_mode='go_above_table').add([go_above_table, joint_switch, waiting])

    def execute(self, userdata):
        self.call_ha(self.return_ha.xml())
        return 'done'

class WaitForHAMState(smach.State):
    def __init__(self, desired_ham_state):
        smach.State.__init__(self, outcomes=['done'])
        
        self.current_ham_state = "Undefined"
        self.desired_ham_state = desired_ham_state
        rospy.Subscriber("/ham_state", HAMState, self.ham_callback)
    
    def ham_callback(self, msg):
        self.current_ham_state = msg.executing_control_mode_name
    
    def execute(self, userdata):
        self.current_ham_state = "Undefined"
        
        while self.current_ham_state != self.desired_ham_state:
            rospy.sleep(0.3)
        
        return 'done'

def main():
    parser = argparse.ArgumentParser(description='Record sliding wall grasps.')
    
    parser.add_argument('directory', type=str, default='foo',
                        help='the directory to write files to')
    parser.add_argument('--angle', type=float, default = 10.,
                        help='angle of attack during sliding (in deg)  (default: 10.0)')
    parser.add_argument('--speed', type=float, default = 0.01,
                        help='speed during sliding  (default: 0.01)')
    parser.add_argument('--inflation', type=float, default = 0.05,
                        help='inflation of fingers during sliding  (default: 0.05)')
    parser.add_argument('--force', type=float, default = 4.,
                        help='downward force during sliding  (default: 4.0)')
    parser.add_argument('--wallforce', type=float, default = -6.,
                        help='wall force to finish sliding  (default: -6.0)')
    #parser.add_argument('--label', type=str, default="foo",
    #                    help='the object label to start with')
    #parser.add_argument('--num', type=int, default=-1,
    #                    help='the trial number to start with')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print "Directory '%s' does not exist!" % (args.directory)
        sys.exit()
    
    experiment_label = datetime.datetime.now().strftime("%Y%b%d_%H-%M-%S")
    
    rospy.init_node('top_down_grasps_with_different_shapes')

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['FINISHED'])
    sm.userdata.trial_label = experiment_label #args.label
    sm.userdata.trial_number = 0 #args.num
    sm.userdata.pid = []
    
    experimental_parameters = {}
    #[math.radians(x) for x in [0, 10., 20., 30., 40.]]
    experimental_parameters['angle_of_attack'] = math.radians(args.angle)
    experimental_parameters['sliding_speed'] = args.speed
    experimental_parameters['finger_inflation'] = args.inflation
    experimental_parameters['downward_force'] = args.force
    experimental_parameters['wall_force'] = args.wallforce
    
    with open(args.directory + '/' + experiment_label + '.yml', 'w') as outfile:
        outfile.write( yaml.dump(experimental_parameters, default_flow_style=False) )
    
    # Open the container
    with sm:
        # Add states to the container
        #smach.StateMachine.add('LABEL_TRIAL', LabelTrial(max_trials = len(angles_of_attack)),
        #                       transitions={'next_trial':'START_BAG_RECORDING', 'quit':'FINISHED'},
        #                       remapping={'label_in':'trial_label', 
        #                                  'label_out':'trial_label',
        #                                  'counter_in':'trial_number',
        #                                  'counter_out':'trial_number'})
        smach.StateMachine.add('RETURN_TO_START', ReturnToStart(experimental_parameters), 
                               transitions={'done':'WAIT_FOR_HAM_STATE_INIT'})
        smach.StateMachine.add('WAIT_FOR_HAM_STATE_INIT', WaitForHAMState('waiting'),
                               transitions={'done':'START_BAG_RECORDING'})
        smach.StateMachine.add('START_BAG_RECORDING', StartBagRecording(["/ee_velocity", "/joint_states", "/pressures", "/tf", "/ft_sensor", "/ft_world", "/ham_state"], prefix = args.directory + '/'),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number',
                                          'pid_in':'pid',
                                          'pid_out':'pid'},
                               transitions={'done':'START_VIDEO_RECORDING'})
        smach.StateMachine.add('START_VIDEO_RECORDING', StartBagRecording(["/camera1/image_color", "/camera2/image_color"], prefix = args.directory + "/", suffix = "_video"),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number',
                                          'pid_in':'pid',
                                          'pid_out':'pid'},
                               transitions={'done':'EXECUTE_MOTION'})
        smach.StateMachine.add('EXECUTE_MOTION', ExecuteMotion(experimental_parameters),
                               remapping={'label_in':'trial_label',
                                          'counter_in':'trial_number'},
                               transitions={'done':'WAIT_FOR_HAM_STATE'})
        smach.StateMachine.add('WAIT_FOR_HAM_STATE', WaitForHAMState('finished'),
                               transitions={'done':'STOP_BAG_RECORDINGS'})
        smach.StateMachine.add('STOP_BAG_RECORDINGS', KillPG(),
                               remapping={'pid_in':'pid',
                                          'pid_out':'pid'},
                               transitions={'done':'FINISHED'})
    
    sm.set_initial_state(['RETURN_TO_START'])
    
    # Execute SMACH plan
    outcome = sm.execute()


if __name__ == '__main__':
    main()
