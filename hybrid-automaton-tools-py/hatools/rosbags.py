import rosbag

class BagJointTrajectory:
    
    def __init__(self, bag_filename):
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_timestamps = []
        
        self.valve_commands = []
        self.valve_timestamps = []
        
        with rosbag.Bag(bag_filename, 'r') as bag:
            cnt = 0
            for topic, msg, t in bag.read_messages(['/joint_states']):
                if cnt == 0:
                    initial_timestamp = msg.header.stamp
                
                self.joint_positions.append(msg.position)
                self.joint_velocities.append(msg.velocity)
                self.joint_timestamps.append(msg.header.stamp - initial_timestamp)
                
                cnt = cnt + 1
            
            rospy.loginfo("Generated joint trajectory with {} points.".format(cnt))
            
            self.joint_positions = np.array(self.joint_positions)
            self.joint_velocities = np.array(self.joint_velocities)
            self.joint_timestamps = np.array(self.joint_timestamps).reshape(-1, 1)
            
            cnt = 0
            for topic, msg, t in bag.read_messages(['/softhand/commands']):
                if cnt == 0:
                    initial_timestamp = t
                
                self.valve_commands.append(msg.data)
                self.valve_timestamps.append([t - initial_timestamp] * len(msg.data))
                
                cnt = cnt + 1
                        
            rospy.loginfo("Generated valve trajectory with {} points.".format(len(self.valve_commands)))
    
            self.valve_commands = np.array(self.valve_commands)
            self.valve_timestamps = np.array(self.valve_timestamp)
    
    def rospy_durations_to_string(self, a):
        tmp = "[%i,%i]" % (a.shape[0], a.shape[1])
        tmp += ';'.join([str(d) for d in [','.join([str(e.to_sec()) for e in a[i]]) for i in range(a.shape[0])]])
        return tmp
    
    def get_ha_string(self, controllers):
        return '<HybridAutomaton name="HA" current_control_mode="MyCM"> <ControlMode name="MyCM"> <ControlSet type="rxControlSet" name=""> ' + ' '.join(controllers) + '</ControlSet></ControlMode></HybridAutomaton>'
    
    def get_softhand_controller_string(self, **kwargs):
        timestamps = self.rospy_durations_to_string(self.valve_timestamps.T)
        if (kwargs.has_key('dt')):
            init_time = 0
            if (kwargs.has_key('initial_time')):
                init_time = kwargs['initial_time']
            timestamps = self.matrix_to_string(np.arange(self.valve_timestamps.T.shape[1]) * np.ones(self.valve_timestamps.T.shape) * kwargs['dt'] + init_time)
        tmp = '<Controller type="SoftHandController" name="GraspControl" goal="%s" goal_is_relative="1" valve_times="%s"/>' % (self.matrix_to_string(self.valve_commands.T), timestamps)
        return tmp

    def get_ha_valve_string(self, **kwargs):
        #delta_t = np.diff(self.valve_timestamps[:,0])
        #print [d.to_sec() for d in delta_t]
        #print ' '.join([d.to_sec() for d in delta_t])
        return self.get_ha_string([self.get_softhand_controller_string(**kwargs)])
    
    def get_arm_controller_string(self, **kwargs):
        nth = 10
        initial_time = 1.
        if (kwargs.has_key('initial_time')):
            initial_time = kwargs['initial_time']
        if (kwargs.has_key('nth')):
            nth = kwargs['nth']
        delta_t = np.vstack([rospy.Duration(initial_time), np.diff(self.joint_timestamps[::nth], axis=0)])
        assert(len(self.joint_positions[::nth]) == len(delta_t))
        
        traj = self.joint_positions[::nth].T
        if (kwargs.has_key('noise')):
            traj += kwargs['noise'][::nth].T
        
        tmp = '<Controller type="InterpolatedJointController" name="HomeCtrl" goal_is_relative="0" kp="[7,1]300;200;150;20;10;10;10" kv="[7,1]2;4;2;0.8;0.2;0.2;0.02" goal="%s" completion_times="%s" v_max="[0,0]" a_max="[0,0]" interpolation_type="linear" />' % (self.matrix_to_string(traj), self.rospy_durations_to_string(delta_t))
        return tmp
    
    def get_arm_string(self, nth=1, hand_time=[9.8]*4, finger_openings=[3.5]*4, initial_time=1.0):
        assert(len(hand_time) == len(finger_openings))        
        hand = '<Controller type="SoftHandController" name="GraspControl" goal="[4,1]%s" goal_is_relative="1" valve_times="[4,1]%s"/>' % (';'.join([str(x) for x in finger_openings]), ';'.join([str(x) for x in hand_time]))

        return self.get_ha_string([self.get_arm_controller_string({'nth': nth, 'initial_time': initial_time}), hand])
    
    def get_arm_softhand_string(self, **kwargs):
        return self.get_ha_string([self.get_arm_controller_string(**kwargs), self.get_softhand_controller_string(**kwargs)])
