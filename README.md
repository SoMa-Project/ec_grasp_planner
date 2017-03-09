# Grasp Planner based on Environmental Constraint Exploitation

## Background
This is based on:

Clemens Eppner and Oliver Brock. "[Planning Grasp Strategies That Exploit Environmental Constraints](http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/eppner_icra2015.pdf)"  
Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pp. 4947 - 4952, 2015.

## Install

```
catkin build ec_grasp_planner
```

### Build Dependencies

```
rosdep install ec_grasp_planner
```

### Runtime Dependencies

- ecto_rbo


<!--
## Structure and Flow of Information

<img src="docs/diagram.png" alt="Diagram" width="200" />
-->

## Usage

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

## Examples

```
roslaunch iiwa_gazebo iiwa_gazebo.launch
roslaunch trik_controller iiwa.launch
rosrun ecto_rbo_yaml plasm_yaml_ros_node.py demo_vision.yaml --debug

rosrun ec_grasp_planner planner.py --grasp surface_grasp --ros_service_call
```