#!/usr/bin/env python

import math
import numpy as np
from tf import transformations as tra

# python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp wall_grasp wall_chewinggum
# python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp edge_grasp --edgedistance -0.007 edge_chewinggum/
# python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp surface_grasp test_folder

class BaseHandArm(dict):
    def __init__(self):
        self['mesh_file'] = "Unknown"
        self['mesh_file_scale'] = 1.
        
        self['wall_grasp'] = {}
        self['edge_grasp'] = {}
        self['surface_grasp'] = {}

class RBOHand2(BaseHandArm):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1


class RBOHand2WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2WAM, self).__init__()
        
        self['surface_grasp']['initial_goal'] = np.array([0.910306, -0.870773, -2.36991, 2.23058, -0.547684, -0.989835, 0.307618])
        self['surface_grasp']['pose'] = tra.translation_matrix([0, 0, 0])
        self['surface_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.2])
        self['surface_grasp']['downward_force'] = 7.
        self['surface_grasp']['valve_pattern'] = (np.array([[ 0. ,  4.1], [ 0. ,  0.1], [ 0. ,  5. ], [ 0. ,  5.], [ 0. ,  2.], [ 0. ,  3.5]]), np.array([[1,0]]*6))
        
        self['wall_grasp']['table_force'] = 3.
        self['wall_grasp']['sliding_speed'] = 0.04
        self['wall_grasp']['wall_force'] = -11.0
        self['wall_grasp']['angle_of_attack'] = math.radians(69.0)
        
        self['edge_grasp']['edge_distance_factor'] = -0.007
        self['edge_grasp']['distance'] = 0.
        self['edge_grasp']['downward_force'] = 4.0
        self['edge_grasp']['sliding_speed'] = 0.04
        self['edge_grasp']['angle_of_sliding'] = math.radians(-10.)

class RBOHand2Kuka(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2WAM, self).__init__()
        
        self['surface_grasp']['initial_goal'] = np.array([0, 0, 0, 0, 0, 0])
        self['surface_grasp']['pose'] = tra.translation_matrix([0, 0, 0])
        self['surface_grasp']['pregrasp_pose'] = tra.translation_matrix([0, 0, -0.2])
        self['surface_grasp']['downward_force'] = 7.
        self['surface_grasp']['valve_pattern'] = (np.array([[ 0. ,  4.1], [ 0. ,  0.1], [ 0. ,  5. ], [ 0. ,  5.], [ 0. ,  2.], [ 0. ,  3.5]]), np.array([[1,0]]*6))
        
        self['wall_grasp']['table_force'] = 3.
        self['wall_grasp']['sliding_speed'] = 0.04
        self['wall_grasp']['wall_force'] = -11.0
        self['wall_grasp']['angle_of_attack'] = math.radians(69.0)
        
        self['edge_grasp']['edge_distance_factor'] = -0.007
        self['edge_grasp']['distance'] = 0.
        self['edge_grasp']['downward_force'] = 4.0
        self['edge_grasp']['sliding_speed'] = 0.04
        self['edge_grasp']['angle_of_sliding'] = math.radians(-10.)
        
        