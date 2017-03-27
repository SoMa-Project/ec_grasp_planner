# Grasp Planner based on Environmental Constraint Exploitation

## Table of Contents

1. [Overview](#overview)
2. [Install](#install)
   1. [Minimal Dependencies](#minimaldependencies)
   2. [Dependencies For Running the Gazebo Example](#gazebodependencies)
   3. [Grasp Planner](#planner)
3. [Usage](#usage)
4. [Examples](#examples)
   1. [Planning Based on PCD Input](#example1)
   2. [Planning Based on Continuous RGB-D Input](#example2)
   3. [Kuka Arm in Gazebo Simulation with TRIK Controller](#example3)

---

## Overview <a name="overview"></a>

<img src="docs/example1.png" alt="Diagram" width="300" />

This is based on:

Clemens Eppner and Oliver Brock. "[Planning Grasp Strategies That Exploit Environmental Constraints](http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/eppner_icra2015.pdf)"  
Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pp. 4947 - 4952, 2015.

<!--
### Structure and Flow of Information

<img src="docs/diagram.png" alt="Diagram" width="200" />
-->

---

## Install <a name="install"></a>

This code was tested with [ROS indigo](http://wiki.ros.org/indigo) under Ubuntu 14.04.5 (LTS).

### Minimal Dependencies <a name="minimaldependencies"></a>

<!--
```
rosdep install ec_grasp_planner
```
-->

* Clone the ROS stack [ecto_rbo](https://github.com/SoMa-Project/vision.git) in your catkin workspace and build it:
```
git clone https://github.com/SoMa-Project/vision.git
catkin build ecto_rbo
```

* Get [PyDDL](https://github.com/garydoranjr/pyddl):
```
pip install -e git+https://github.com/garydoranjr/pyddl.git#egg=pyddl
```

* Get the ROS package hybrid_automaton_msgs from [hybrid_automaton_manager_kuka](https://github.com/SoMa-Project/hybrid_automaton_manager_kuka.git):
```
git clone https://github.com/SoMa-Project/hybrid_automaton_manager_kuka.git
```

### Dependencies For Running the Gazebo Example <a name="gazebodependencies"></a>

* Get Gazebo multi-robot simulator, version 2.2.6:
```
  sudo apt-get install ros-indigo-gazebo-*
```

* Get [iiwa_stack](https://github.com/SalvoVirga/iiwa_stack.git):
```
  git clone https://github.com/SalvoVirga/iiwa_stack.git
  cd iiwa_stack
  git checkout 94670d70b9bfbf0920c7de539012c805734fdbc5
  catkin build iiwa
```


* Get [hybrid_automaton_library](https://github.com/tu-rbo/hybrid-automaton-library.git) and install it by following the readme instructions.


* Before you continue, please make sure that you have installed a TRIK controller, e.g. [trik_controller](https://github.com/SoMa-Project/trik_controller.git).



* Build the [hybrid_automaton_manager_kuka](https://github.com/SoMa-Project/hybrid_automaton_manager_kuka.git) and link robot files from iiwa_stack:
```
  catkin build hybrid_automaton_manager_kuka
  IIWA_STACK=`rospack find iiwa_description`
  HA_MANAGER=`rospack find hybrid_automaton_manager_kuka`
  ln -s $HA_MANAGER/../iiwa_description/launch/iiwa7_kinect_ft_upload.launch_ $IIWA_STACK/launch/iiwa7_kinect_ft_upload.launch
  ln -s $HA_MANAGER/../iiwa_description/urdf/iiwa7_kinect_ft.xacro_ $IIWA_STACK/urdf/iiwa7_kinect_ft.xacro
  ln -s $HA_MANAGER/../iiwa_description/urdf/iiwa7_kinect_ft.urdf.xacro_ $IIWA_STACK/urdf/iiwa7_kinect_ft.urdf.xacro
```

### Grasp Planner <a name="planner"></a>

Now, you can clone this repository into your catkin workspace and build the ROS package:

```
catkin clone https://github.com/SoMa-Project/ec_grasp_planner.git
cd ec_grasp_planner
git submodule init
git submodule update
catkin build ec_grasp_planner
```


---

## Usage <a name="usage"></a>

```
planner.py [-h] [--ros_service_call] [--file_output]
                [--grasp {any,EdgeGrasp,WallGrasp,SurfaceGrasp}]
                [--grasp_id GRASP_ID] [--rviz]
                [--robot_base_frame ROBOT_BASE_FRAME]
                [--object_frame OBJECT_FRAME] [--handarm HANDARM]

Turn path in graph into hybrid automaton.

optional arguments:
  -h, --help            show this help message and exit
  --ros_service_call    Whether to send the hybrid automaton to a ROS service
                        called /update_hybrid_automaton. (default: False)
  --file_output         Whether to write the hybrid automaton to a file called
                        hybrid_automaton.xml. (default: False)
  --grasp {any,EdgeGrasp,WallGrasp,SurfaceGrasp}
                        Which grasp type to use. (default: any)
  --grasp_id GRASP_ID   Which specific grasp to use. Ignores any values < 0.
                        (default: -1)
  --rviz                Whether to send marker messages that can be seen in
                        RViz and represent the chosen grasping motion.
                        (default: False)
  --robot_base_frame ROBOT_BASE_FRAME
                        Name of the robot base frame. (default: world)
  --object_frame OBJECT_FRAME
                        Name of the object frame. (default: object)
  --handarm HANDARM     Python class that contains configuration parameters
                        for hand and arm-specific properties. (default:
                        RBOHand2WAM)

```

---

## Examples  <a name="examples"></a>

### Planning Based on PCD Input  <a name="example1"></a>

This example shows a planned grasp in RViz based on a PCD file.

```
roscore

rosrun ecto_rbo_yaml plasm_yaml_ros_node.py `rospack find ec_grasp_planner`/data/geometry_graph_example1.yaml --debug

# start visualization
rosrun rviz rviz -d `rospack find ec_grasp_planner`/configs/ec_grasps_example1.rviz

# select which type of grasp you want
rosrun ec_grasp_planner planner.py --rviz --robot_base_frame camera_rgb_optical_frame --grasp WallGrasp 
```

In RViz you should be able to see the geometry graph:

<img src="docs/example1.png" alt="Diagram" width="300" />

### Planning Based on Continuous RGB-D Input   <a name="example2"></a>

This example shows how to use the planner with an RGB-Depth sensor like Kinect or Asus Xtion.

```
roslaunch openni2_launch openni2.launch depth_registration:=true
rosrun dynamic_reconfigure dynparam set /camera/driver depth_mode 5

rosrun ecto_rbo_yaml plasm_yaml_ros_node.py `rospack find ec_grasp_planner`/data/geometry_graph_example2.yaml --debug

rosrun rviz rviz -d `rospack find ec_grasp_planner`/configs/ec_grasps_example2.rviz

rosrun ec_grasp_planner planner.py --grasp EdgeGrasp --rviz
```

### Kuka Arm in Gazebo Simulation with TRIK Controller  <a name="example3"></a>

This example shows the execution of a planned hybrid automaton motion.

```
roscore
rosparam set use_sim_time 1
roslaunch iiwa_gazebo iiwa_gazebo_examples.launch model:=iiwa7_kinect_ft world:=iiwa_ex3
roslaunch trik_controller iiwa.launch
rosservice call /disable
rosrun hybrid_automaton_manager_kuka hybrid_automaton_manager_kuka

rosrun ecto_rbo_yaml plasm_yaml_ros_node.py `rospack find ec_grasp_planner`/data/geometry_graph_example3.yaml --debug

# to check the potential grasps
rosrun rviz rviz -d `rospack find ec_grasp_planner`/configs/ec_grasps_example3.rviz

rosrun ec_grasp_planner planner.py --grasp SurfaceGrasp --ros_service_call --rviz --handarm RBOHand2Kuka
```

***
