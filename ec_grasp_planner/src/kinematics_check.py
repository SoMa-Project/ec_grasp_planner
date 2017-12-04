#!/usr/bin/env python

# Copyright 2016-2017 Robotics and Biology Lab, TU Berlin. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
#     Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#     Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.

# @author Can Erdogan
# @date 2017-12-04
# @brief Wraps the robotics library interface to check kinematics for a given pose. Provides a service call with returns 1 for collision and 0 for free.


from ec_grasp_planner import srv
import rospy
import subprocess
import rospkg
import time

def handle_add_two_ints(req):

    # Make call to the kinematic check program in roblib
    jointStr = ",".join(format(x, "f") for x in req.joints)
    rospack = rospkg.RosPack()
    this_pkg_path = rospack.get_path('ec_grasp_planner')
    roblib_pkg_path = this_pkg_path + '/../../contact-motion-planning/'
    subprocess.Popen([roblib_pkg_path+'build/demos/rlSomaDemo/rlSomaDemod', '--rootDir', roblib_pkg_path, '--joints', jointStr])
    time.sleep(1)
    
    # Read the output that is written to a file
    file = open(this_pkg_path + '/src/collision-result.txt', 'r')
    val = file.readlines()
    print val
    print int(val[0])
        

    return srv.CheckKinematicsResponse(req.joints[0] + req.joints[1])

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', srv.CheckKinematics, handle_add_two_ints)
    print "Ready to add two ints."
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()
