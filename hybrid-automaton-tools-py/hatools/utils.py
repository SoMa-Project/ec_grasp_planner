import numpy as np

def msg_to_matrix(tf_pose_msg):
    import tf_conversions.posemath as pm
    return pm.toMatrix(pm.fromMsg(tf_pose_msg))

def matrix_to_string(m):
    a = m.reshape((-1, 1)) if (len(m.shape) == 1) else m
    tmp = "[%i,%i]" % (a.shape[0], a.shape[1]) if a.size > 0 else "[0,0]"
    tmp += ';'.join([d for d in [','.join(['{:f}'.format(e) for e in a[i]]) for i in range(a.shape[0])]])
    return tmp

def string_to_matrix(s):
    s = s.strip()
    sep_brack = s.find(']')
    shape = (int(s[1:s.find(',')]), int(s[s.find(',')+1:sep_brack]))
    m = np.zeros(shape)
    if shape == (0, 0):
        return m
    
    for i, row in enumerate(s[sep_brack+1:].split(';')):
        for j, col in enumerate(row.split(',')):
            m[i, j] = float(col)
    if shape[1] == 1:
        m = m.reshape(-1,)
    return m

def string_is_matrix(s):
    return s.strip()[0] == "["

def numpify_dict(d):
    for key, value in d.iteritems():
        if isinstance(value, str) or isinstance(value, unicode): # with Python3 this might make problems
            if string_is_matrix(value):
                d[key] = string_to_matrix(value)
    return d

def dict_to_string(d):
    strings = []
    for key, value in d.iteritems():
        if not isinstance(value, np.ndarray) and value == '':  # added the isinstance condition due to "FutureWarning: elementwise comparison failed"
            continue
        strings.append(str(key) + '="' + (matrix_to_string(value) if isinstance(value, np.ndarray) else str(value)) + '"')
    return ' '.join(strings)

def generate_noise(dU, smooth=True, var=1.0, renorm=False):
    """
    Generate a T x dU gaussian-distributed noise vector.
    This will approximately have mean 0 and variance 1, including smoothing.

    Args:
        T (int): # Timesteps
        dU (int): Dimension of actions
        smooth (bool, optional): Perform smoothing of noise.
        var (float, optional): If smooth=True, applies a gaussian filter with this variance.
        renorm (bool, optional): If smooth=True, renormalizes data to have variance 1 after smoothing.

    Sanity Check
    >>> np.random.seed(123)
    >>> generate_noise(5, 2)
    array([[-1.0856306 ,  0.99734545],
           [ 0.2829785 , -1.50629471],
           [-0.57860025,  1.65143654],
           [-2.42667924, -0.42891263],
           [ 1.26593626, -0.8667404 ]])
    >>> np.random.seed(123)
    >>> generate_noise(5, 2, smooth=True, var=0.5)
    array([[-0.93944619,  0.73034299],
           [ 0.04449717, -0.90269245],
           [-0.68326104,  1.09300178],
           [-1.8351787 , -0.25446477],
           [ 0.87139343, -0.81935331]])
    """
    import scipy.ndimage as sp_ndimage
    T = self.get_length()
    noise = np.random.randn(T, dU)
    if smooth:
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise/np.sqrt(variance)
    return noise


# this is rswin/hybrid_automaton/src/JumpCondition.cpp::_computeJumpCriterion(..)
def compute_jump_criterion(jump, x, y):
    assert(x.shape == y.shape)
    ret = 0
    weights = jump.properties['norm_weights']
    if weights.size == 0:
        weights = np.ones(x.shape)
    
    criterion = jump.properties['jump_criterion']
    
    if criterion == "NORM_L1" or criterion == 0 or criterion == "0":
        ret = np.dot(weights, np.fabs(x - y))
    elif criterion == "NORM_L2" or criterion == 1 or criterion == "1":
        ret = np.sqrt(np.dot(weights, np.power(np.fabs(x - y), 2)))   # i know, it's redundant but let's stay as close as possible to the original cpp-source
    elif criterion == "NORM_L_INF" or criterion == 2 or criterion == "2":
        ret = np.max(weights * np.fabs(x - y))
    elif criterion == "NORM_ROTATION" or criterion == 3 or criterion == "3":
        assert(x.shape == (3, 3) and y.shape == (3, 3))
        x_rot = np.eye(4)
        x_rot[:3,:3] = np.dot(x.T, y)
        ret, _, _ = tra.rotation_from_matrix(x_rot)
    elif criterion == "NORM_TRANSFORM" or criterion == 4 or criterion == "4":
        assert(x.shape == (4, 4) and y.shape == (4, 4) and weights.shape == (2,))
        x_rot = np.dot(tra.inverse_matrix(x), y)
        angle_diff, _, _ = tra.rotation_from_matrix(x_rot)
        dist_diff = np.linalg.norm(tra.translation_from_matrix(x) - tra.translation_from_matrix(y))
        ret = np.dot(weights, [angle_diff, dist_diff])
    elif criterion == "THRESH_UPPER_BOUND" or criterion == 5 or criterion == "5":
        ret = np.max(weights * (y - x))
    elif criterion == "THRESH_LOWER_BOUND" or criterion == 6 or criterion == "6":
        ret = np.max(weights * (x - y))
    
    return ret

# this is rswin/hybrid_automaton/src/JumpCondition.cpp::isActive()
def jump_is_active(jump, sensor_value):
    goal_is_relative = (jump.properties['goal_is_relative'] == '1' or jump.properties['goal_is_relative'] == True)
    if jump.properties.has_key('goal'):
        desired = jump.properties['goal']
    else:
        pass
        # get goal from the controller
        #jump.properties['controller'] 
    negate = (jump.properties['negate'] == '1' or jump.properties['negate'] == True)
    epsilon = float(jump.properties['epsilon'])
    
    if goal_is_relative:
        if not negate:
            return compute_jump_criterion(jump, sensor_value, desired) <= epsilon
        else:
            return compute_jump_criterion(jump, sensor_value, desired) > epsilon
    else:
        if not negate:
            return compute_jump_criterion(jump, sensor_value, desired) <= epsilon
        else:
            return compute_jump_criterion(jump, sensor_value, desired) > epsilon

def evalute_switch(switch, sensor_values):
    ret = True
    for i, jump in enumerate(switch.conditions):
        ret = ret and jump_is_active(jump, sensor_values[i])
    return ret
