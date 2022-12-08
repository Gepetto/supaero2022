'''
Load 4 times the UR5 model, plus a plate object on top of them, to feature a simple parallel robot.
No optimization, this file is just an example of how to load the models.
'''

import numpy as np
import pinocchio as pin
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
# %jupyter_snippet 0
from utils.load_ur5_parallel import load_ur5_parallel

# Load 4 Ur5 robots, placed at 0.3m from origin in the 4 directions x,y,-x,-y.
robot = load_ur5_parallel()

# Open the viewer
viz = MeshcatVisualizer(robot)
# %end_jupyter_snippet

viz.display(robot.q0)

# %jupyter_snippet 1
# Add a new object featuring the parallel robot tool plate.
[w, h, d] = [0.5, 0.5, 0.005]
color = [red, green, blue, transparency] = [1, 1, 0.78, .3]
viz.addBox('world/robot0/toolplate', [w, h, d], color)
Mtool = pin.SE3(pin.utils.rotate('z', 1.268), np.array([0, 0, .75]))
viz.applyConfiguration('world/robot0/toolplate', Mtool)
viz.display(robot.q0)
# %end_jupyter_snippet

# The 4 effector placements are computed by:
# %jupyter_snippet 2
effector_indexes = [robot.model.getFrameId('tool0_#%d' % i) for i in range(4)]
robot.framePlacement(robot.q0, effector_indexes[0])
# %end_jupyter_snippet
