# Grasp Planner based on Environmental Constraint Exploitation

1. [Background](#background)
2. [Install](#install)
3. [Usage](#usage)
4. [Examples](#examples)

---

## Background <a name="background"></a>
This is based on:

Clemens Eppner and Oliver Brock. "[Planning Grasp Strategies That Exploit Environmental Constraints](http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/eppner_icra2015.pdf)"  
Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pp. 4947 - 4952, 2015.

### Structure and Flow of Information

<img src="docs/diagram.png" alt="Diagram" width="200" />


---

## Install <a name="install"></a>

```
catkin build ec_grasp_planner
```

### Build-Time Dependencies

#### General
```
rosdep install ec_grasp_planner
```


### Runtime Dependencies

- ecto_rbo

#### For Running the Gazebo Example

- Get Gazebo multi-robot simulator, version 2.2.6:
```
  sudo apt-get install ros-indigo-gazebo-*
```

- Get iiwa_stack:
```
  git clone https://github.com/SalvoVirga/iiwa_stack.git
  roscd iiwa/..
  git checkout 94670d70b9bfbf0920c7de539012c805734fdbc5
  catkin build iiwa
```

- Get soma_wp4_5_examples and link robot files from iiwa_stack:
```
  git clone https://github.com/SoMa-Project/soma_wp4_5_examples.git
  catkin build ex8_HA_manager
  IIWA_STACK=`rospack find iiwa_description`
  HA_MANAGER=`rospack find ex8_HA_manager`
  ln -s $HA_MANAGER/iiwa_description/launch/iiwa7_kinect_ft_upload.launch_ $IIWA_STACK/launch/iiwa7_kinect_ft_upload.launch
  ln -s $HA_MANAGER/iiwa_description/urdf/iiwa7_kinect_ft.xacro_ $IIWA_STACK/urdf/iiwa7_kinect_ft.xacro
  ln -s $HA_MANAGER/iiwa_description/urdf/iiwa7_kinect_ft.urdf.xacro_ $IIWA_STACK/urdf/iiwa7_kinect_ft.urdf.xacro
```

---

## Usage <a name="usage"></a>

```
planner.py [-h] [--ros_service_call] [--file_output]
                  [--grasp {any,edge_grasp,wall_grasp,surface_grasp}]

Execute grasp strategy by turning a path in the geometry graph into a hybrid automaton.

optional arguments:
  -h, --help            show this help message and exit
  --ros_service_call    Whether to send the hybrid automaton to a ROS service
                        called /update_hybrid_automaton. (default: False)
  --file_output         Whether to write the hybrid automaton to a file called
                        hybrid_automaton.xml. (default: False)
  --grasp {any,edge_grasp,wall_grasp,surface_grasp}
                        which grasp type to use (default: any)
```

---

## Examples  <a name="examples"></a>

### Kuka Arm in Gazebo simulation with TRIK controller
```
# ensure rosparam set use_sim_time 1
roslaunch iiwa_gazebo iiwa_gazebo.launch model:=iiwa7_kinect_ft
roslaunch trik_controller iiwa.launch

rosrun ecto_rbo_yaml plasm_yaml_ros_node.py demo_vision.yaml --debug

# to check the potential grasps
rosrun rviz rviz -d `rospack find ec_grasp_planner`/configs/ec_grasps.rviz

rosrun ec_grasp_planner planner.py --grasp surface_grasp --ros_service_call
```

***

## TODOs

[ ] How to modify hand-specific information