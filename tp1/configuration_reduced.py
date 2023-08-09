"""
Stand-alone program to optimize the configuration q=[q1,q2] of a 2-R robot with
scipy BFGS.
"""

# %jupyter_snippet import
import time

import numpy as np
from scipy.optimize import fmin_bfgs

from utils.meshcat_viewer_wrapper import MeshcatVisualizer, planar, translation2d

# %end_jupyter_snippet

viz = MeshcatVisualizer(url="classical")

# %jupyter_snippet create
viz.addSphere("joint1", 0.1, [1, 0, 0, 1])
viz.addSphere("joint2", 0.1, [1, 0, 0, 1])
viz.addSphere("joint3", 0.1, [1, 0, 0, 1])
viz.addCylinder("arm1", 0.75, 0.05, [0.65, 0.65, 0.65, 1])
viz.addCylinder("arm2", 0.75, 0.05, [0.65, 0.65, 0.65, 1])
viz.addSphere("target", 0.1001, [0, 0.8, 0.1, 1])
# %end_jupyter_snippet


# %jupyter_snippet display
def display(q):
    """Display the robot in Gepetto Viewer."""
    assert q.shape == (2,)
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[0] + q[1])
    s1 = np.sin(q[0] + q[1])
    viz.applyConfiguration("joint1", planar(0, 0, 0))
    viz.applyConfiguration("arm1", planar(c0 / 2, s0 / 2, q[0]))
    viz.applyConfiguration("joint2", planar(c0, s0, q[0]))
    viz.applyConfiguration("arm2", planar(c0 + c1 / 2, s0 + s1 / 2, q[0] + q[1]))
    viz.applyConfiguration("joint3", planar(c0 + c1, s0 + s1, q[0] + q[1]))


# %end_jupyter_snippet


# %jupyter_snippet endeffector
def endeffector(q):
    """Return the 2D position of the end effector of the robot at configuration q."""
    assert q.shape == (2,)
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[0] + q[1])
    s1 = np.sin(q[0] + q[1])
    return np.array([c0 + c1, s0 + s1])


# %end_jupyter_snippet

# %jupyter_snippet cost
target = np.array([0.5, 0.5])
viz.applyConfiguration("target", translation2d(target[0], target[1]))


def cost(q):
    eff = endeffector(q)
    return np.linalg.norm(eff - target) ** 2


# %end_jupyter_snippet


# %jupyter_snippet callback
def callback(q):
    display(q)
    time.sleep(0.5)


# %end_jupyter_snippet

# %jupyter_snippet optim
q0 = np.array([0.0, 0.0])
qopt_bfgs = fmin_bfgs(cost, q0, callback=callback)
print("\n *** Optimal configuration from BFGS = %s \n\n\n\n" % qopt_bfgs)
# %end_jupyter_snippet
