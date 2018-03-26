Collection of python scripts and modules to easily synthesize robot motion in the form of hybrid automata.
```bash
cp 50-import-ha-stuff.ipy ~/.ipython/profile_default/startup/
rosrun rosh rosh
```

To call ros service that sends HA to robot (without rosh, i.e. in rosh you can do services.update_hybrid_automaton(..)):

```python
import rospy
from hybrid_automaton_msgs import srv
call = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
```
