'''
Stand-alone inverse geom in 3D.  Given a reference translation <target> ,
it computes the configuration of the UR5 so that the end-effector position (3D)
matches the target. This is done using BFGS solver. While iterating to compute
the optimal configuration, the script also display the successive candidate
solution, hence moving the robot from the initial guess configuaration to the
reference target.
'''

import time
import unittest

import numpy as np
import pinocchio as pin
import example_robot_data as robex
from scipy.optimize import fmin_bfgs
from numpy.linalg import norm

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# --- Load robot model
robot = robex.load('solo12')
NQ = robot.model.nq
NV = robot.model.nv

# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

# %jupyter_snippet 1
robot.feetIndexes = [robot.model.getFrameId(frameName) for frameName in ['HR_FOOT', 'HL_FOOT', 'FR_FOOT', 'FL_FOOT']]

# --- Add box to represent target
colors = ['red', 'blue', 'green', 'magenta']
for color in colors:
    viz.addSphere("world/%s" % color, .05, color)
    viz.addSphere("world/%s_des" % color, .05, color)

#
# OPTIM 6D #########################################################
#

targets = [
    np.array([-0.7, -0.2, 1.2]),
    np.array([-0.3, 0.5, 0.8]),
    np.array([0.3, 0.1, -0.1]),
    np.array([0.9, 0.9, 0.5])
]
for i in range(4):
    targets[i][2] += 1


def cost(q):
    '''Compute score from a configuration'''
    cost = 0.
    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i]).translation
        cost += norm(p_i - targets[i])**2
    return cost


def callback(q):
    viz.applyConfiguration('world/box', Mtarget)

    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i])
        viz.applyConfiguration('world/%s' % colors[i], p_i)
        viz.applyConfiguration('world/%s_des' % colors[i], list(targets[i]) + [1, 0, 0, 0])

    viz.display(q)
    time.sleep(1e-2)


Mtarget = pin.SE3(pin.utils.rotate('x', 3.14 / 4), np.array([0.5, 0.1, 0.2]))  # x,y,z
qopt = fmin_bfgs(cost, robot.q0, callback=callback)
# %end_jupyter_snippet


class FloatingTest(unittest.TestCase):
    def test_cost(self):
        self.assertLess(cost(qopt), 1e-10)
