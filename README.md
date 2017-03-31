# Grasp Planner based on Environmental Constraint Exploitation

## Table of Contents

1. [Overview](#overview)
2. [Structure, Interfaces and Flow of Information](#structure)
3. [Install](#install)
   1. [Minimal Dependencies](#minimaldependencies)
   2. [Dependencies For Running the Gazebo Example](#gazebodependencies)
   3. [Grasp Planner](#planner)
4. [Usage](#usage)
5. [Examples](#examples)
   1. [Planning Based on PCD Input](#example1)
   2. [Planning Based on Continuous RGB-D Input](#example2)
   3. [Kuka Arm in Gazebo Simulation with TRIK Controller](#example3)

---

## Overview <a name="overview"></a>

This planning framework generates contact-rich motion sequences to grasp objects.
Within this planning framework, we propose a novel view of grasp planning that centers on the exploitation of environmental contact.
In this view, grasps are sequences of constraint exploitations, i.e. consecutive motions constrained by features in the environment, ending in a grasp.
To be able to generate such grasp plans, it becomes necessary to consider planning, perception, and control as tightly integrated components.
As a result, each of these components can be simplified while still yielding reliable grasping performance.
This implementation is based on:

Clemens Eppner and Oliver Brock. "[Planning Grasp Strategies That Exploit Environmental Constraints](http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/eppner_icra2015.pdf)"  
Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pp. 4947 - 4952, 2015.

---

## Structure, Interfaces and Flow of Information <a name="structure"></a>

This is the structure of the planning framework:

<img src="docs/diagram.png" alt="Diagram" style="margin: auto;" />

It consists of a visual processing component and a planning module.
The visual processing component detects planar surfaces, convex, and concave edges in a point cloud and represents them in a graph structure. 
This component is based on [Ecto](https://plasmodic.github.io/ecto/), a computation graph framework for C++/Python.
The computations are organized as a directed acyclic graph of computing *cells* connected by typed edges.
Single computing *cells* do simple operations such as clustering, segmentation, or fitting models.
The following diagram shows the computation graph for the grasp planning framework: Here, a segmentation soup is generated by different segmentation algorithms based on depth and color. For these segments, planes and edges are fitted. The final output is a geometry graph that describes the spatial structure of the environment.

<img src="docs/ecto_graph.png" alt="Diagram" width="600" align="center" style="margin: auto;" />

The planning module takes this spatial graph as input and combines it with information about the object pose and the type of robotic hand and arm into a planning problem. This planning problem is represented in a STRIPS-like fashion and solved using A<sup>*</sup> search. The output of the planner is a sequence of motions interspersed with contact sensor events.

Summing up, the input to the planning framework is given by:
* **Point Cloud:** This can be provided either by a real RGB-D sensor (see Example 2), a recorded point cloud (see Example 1), or even by a simulated sensor (see Example 3).
* **Object Pose:** This is optional and can also be provided by the Ecto graph computation using a simple heuristic: select the point cluster that is closest to the largest planar surface in the scene.
* **Hand and Robot-specific Information:** This defines how a particular hand slides across a surface, closes its fingers etc. It also includes robot-specific things such as f/t sensor thresholds or velocities. For new hands and/or arms this can be easily extended.

The usual output of a robot motion planner are joint-configuration trajectories.
This planner is different. It outputs so-called hybrid automata. A hybrid automaton is a finite state machine whose states are continuous feedback controllers (based on position, velocity, force, etc.) and transitions are discrete sensor events.
This is because position trajectories lack the expressive power that is needed to capture the feedback-driven contact-rich motions considered here.
Hybrid automata are much more suited in this context. 
As a consequence any entity that wants to execute the generated plans needs to be capable of interpreting those hybrid automata descriptions. We use a [C++ library](https://github.com/tu-rbo/hybrid-automaton-library) that allows serialization/desirialization and can be used to wrap robot-specific interfaces as shown in Example 3.

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

Find path in graph and turn it into a hybrid automaton.

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

This example shows a planned grasp in RViz based on a PCD file that contains a single colored point cloud of a table-top scene with a banana placed in the middle.

```
roscore

# if you want to change which pcd to read, change the file name in the ecto graph yaml
rosrun ecto_rbo_yaml plasm_yaml_ros_node.py `rospack find ec_grasp_planner`/data/geometry_graph_example1.yaml --debug

# start visualization
rosrun rviz rviz -d `rospack find ec_grasp_planner`/configs/ec_grasps_example1.rviz

# select which type of grasp you want
rosrun ec_grasp_planner planner.py --rviz --robot_base_frame camera_rgb_optical_frame --grasp WallGrasp
```

In RViz you should be able to see the geometry graph and the wall grasp published as **visualization_msgs/MarkerArray** under the topic names **geometry_graph_marker** and **planned_grasp_path**:

<img src="docs/example1_graph.png" alt="Graph" width="250" /> <img src="docs/example1_grasp.png" alt="Grasp" width="250" />

### Planning Based on Continuous RGB-D Input   <a name="example2"></a>

This example shows how to use the planner with an RGB-Depth sensor like Kinect or Asus Xtion.
It uses the camera drivers provided in ROS:

```
# plug the camera into your computer
roslaunch openni2_launch openni2.launch depth_registration:=true

# set camera resolution to QVGA
rosrun dynamic_reconfigure dynparam set /camera/driver ir_mode 7
rosrun dynamic_reconfigure dynparam set /camera/driver color_mode 7
rosrun dynamic_reconfigure dynparam set /camera/driver depth_mode 7

rosrun ecto_rbo_yaml plasm_yaml_ros_node.py `rospack find ec_grasp_planner`/data/geometry_graph_example2.yaml --debug

# start visualization
rosrun rviz rviz -d `rospack find ec_grasp_planner`/configs/ec_grasps_example2.rviz

# select an edge grasp and visualize the result in RViz
rosrun ec_grasp_planner planner.py --robot_base_frame camera_rgb_optical_frame --grasp EdgeGrasp --rviz
```

Depending on your input the result in RViz could look like this:

<img src="docs/example2_raw.png" alt="Raw" width="250" /> <img src="docs/example2_graph.png" alt="Graph" width="250" /> <img src="docs/example2_grasp.png" alt="Grasp" width="250" />


### Kuka Arm in Gazebo Simulation with TRIK Controller  <a name="example3"></a>

This example shows the execution of a planned hybrid automaton motion in the Gazebo simulator.

```
# make sure the simulation time is used
roscore
rosparam set use_sim_time 1

# start the simulation environment
roslaunch iiwa_gazebo iiwa_gazebo_examples.launch model:=iiwa7_kinect_ft world:=iiwa_ex3
roslaunch trik_controller iiwa.launch
rosservice call /disable
rosrun hybrid_automaton_manager_kuka hybrid_automaton_manager_kuka

rosrun ecto_rbo_yaml plasm_yaml_ros_node.py `rospack find ec_grasp_planner`/data/geometry_graph_example3.yaml --debug

# to check potential grasps
rosrun rviz rviz -d `rospack find ec_grasp_planner`/configs/ec_grasps.rviz
```

In RViz you should be able to see the point cloud simulated in Gazebo and the geometry graph published as **visualization_msgs/MarkerArray** under the topic name **geometry_graph_marker**:

<img src="docs/example3_gazebo_init.png" alt="Gazebo" width="250" /> <img src="docs/example3_raw.png" alt="Raw" width="250" /> <img src="docs/example3_graph.png" alt="Graph" width="250" />

```
# select a surface grasp, visualize and execute it
rosrun ec_grasp_planner planner.py --grasp SurfaceGrasp --ros_service_call --rviz --handarm RBOHand2Kuka
```

In RViz you should be able to see the planned surface grasp and in Gazebo the robot moves its hand towards the cylinder until contact (https://youtu.be/Q91U9r83Vl0):

<img src="docs/example3_grasp.png" alt="Grasp" width="250" /> <img src="docs/example3_gazebo_final.png" alt="Gazebo" width="250" />
