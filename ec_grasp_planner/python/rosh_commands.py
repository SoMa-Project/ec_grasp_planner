# Send jointspace command
services.update_hybrid_automaton(cookbook.goto_joint(np.array([0,0.2,0,2.3,0,0.5,0]))) 

# Send workspace command
services.enable()
pub = rospy.Publisher('spacenav_pose', msg.geometry_msgs.Twist, queue_size=10) 
for x in range(0,500): 
    pub.publish(msg.geometry_msgs.Twist(linear=msg.geometry_msgs.Vector3(0,0,0.05)))
    sleep(0.01)  
