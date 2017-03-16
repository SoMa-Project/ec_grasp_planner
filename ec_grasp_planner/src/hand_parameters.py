#!/usr/bin/env python

import numpy as np
from tf import transformations as tra
import hatools.components as ha

# python ec_grasps.py --angle 69.0 --inflation .29 --speed 0.04 --force 3. --wallforce -11.0 --positionx 0.0 --grasp wall_grasp wall_chewinggum
# python ec_grasps.py --anglesliding -10.0 --inflation 0.02 --speed 0.04 --force 4.0 --grasp edge_grasp --edgedistance -0.007 edge_chewinggum/
# python ec_grasps.py --anglesliding 0.0 --inflation 0.33 --force 7.0 --grasp surface_grasp test_folder

grasps = {
    'WallGrasp': {
        'jump_condition': ha.ForceTorqueSwitch('slide', 'close_hand', goal = np.array([-10, 0, 0, 0, 0, 0]), norm_weights = np.array([1, 0, 0, 0, 0, 0]), jump_criterion = "THRESH_LOWER_BOUND", frame_id = 'odom', goal_is_relative = '1'),
    },
    'EdgeGrasp': {
    },
    'SurfaceGrasp': {
    },
}
