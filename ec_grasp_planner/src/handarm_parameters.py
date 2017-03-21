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

class RBOHand2(BaseHandArm):
    def __init__(self):
        super(RBOHand2, self).__init__()
        self['mesh_file'] = "package://ec_grasp_planner/data/softhand_right_colored.dae"
        self['mesh_file_scale'] = 0.1


class RBOHand2WAM(RBOHand2):
    def __init__(self, **kwargs):
        super(RBOHand2WAM, self).__init__()
        
        self['surface_grasp']['angle_of_sliding'] = 0
        self['surface_grasp']['downward_force'] = 7.
        
        self['wall_grasp']['table_force'] = 3.
        self['wall_grasp']['sliding_speed'] = 0.04
        self['wall_grasp']['wall_force'] = -11.0
        self['wall_grasp']['angle_of_attack'] = math.radians(69.0)
        
        self['edge_grasp']['edge_distance_factor'] = -0.007
        self['edge_grasp']['distance'] = 0. # TODO
        self['edge_grasp']['downward_force'] = 4.0
        self['edge_grasp']['sliding_speed'] = 0.04
        self['edge_grasp']['angle_of_sliding'] = math.radians(-10.)
        
        
