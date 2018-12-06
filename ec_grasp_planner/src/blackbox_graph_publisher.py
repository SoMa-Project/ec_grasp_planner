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
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from subprocess import call
from hybrid_automaton_msgs import srv
from hybrid_automaton_msgs.msg import HAMState

from std_msgs.msg import Header

from pregrasp_msgs.msg import GraspStrategyArray
from pregrasp_msgs.msg import GraspStrategy

from geometry_graph_msgs.msg import Graph
from geometry_graph_msgs.msg import Node
from geometry_graph_msgs.msg import Edge

from ec_grasp_planner import srv

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

import pyddl

import rospkg
#from bzrlib.lsprof import label
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('ec_grasp_planner')
sys.path.append(pkg_path + '/../hybrid-automaton-tools-py/')
import hatools.components as ha
import hatools.cookbook as cookbook



def talker():
    pub = rospy.Publisher('Subscriber', Graph, queue_size=10)
    rospy.init_node('blackboarGraphPublisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    T_positioning = Pose()
    T_positioning.position = Point(x = 0, y = 0, z = 0.5)
    T_positioning.orientation = Quaternion(x = 0, y = 0, z = 0, w = -1) 
    #tra.concatenate_matrices(tra.translation_matrix([0, 0, 0.5]), tra.rotation_matrix(math.radians(0.0), [0, 0, 1]))
    
    T_surface = Pose()
    T_surface.position = Point(x = 0.119921199977, y = -0.00783250015229, z = 0.789280951023)
    T_surface.orientation = Quaternion(x = -0.663539171219, y = 0.726180911064, z = -0.175170898438, w = -0.0411370396614)
    #tra.concatenate_matrices(tra.translation_matrix([, , ]), tra.transformations.quaternion_matrix([-0.663539171219, 0.726180911064, -0.175170898438, -0.0411370396614]) )
    
    # create nodes
    node1 = Node()
    node2 = Node()    
    node1.transform = T_positioning
    node1.label = "Positioning"    
    node2.transform = T_surface
    node2.label = "SurfaceGrasp"
    
    # create the edge between the two nodes
    edge1  =  Edge()
    edge1.node_id_start = 0 
    edge1.node_id_end = 1
    # put together graph
    graph = Graph()    
    graph.nodes.append(node1)
    graph.nodes.append(node2) 
    graph.edges.append(edge1)
    
    # publish graph
    while not rospy.is_shutdown():    
        #rospy.loginfo(graph)        
        pub.publish(graph)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

