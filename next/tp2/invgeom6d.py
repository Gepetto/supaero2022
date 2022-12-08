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
robot = robex.load('ur5')
NQ = robot.model.nq
NV = robot.model.nv

# Open the viewer
viz = MeshcatVisualizer(robot)

# Define an init config
robot.q0 = np.array([0, -3.14 / 2, 0, 0, 0, 0])
viz.display(robot.q0)
time.sleep(.3)
print("Let's go to pdes.")

# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [.05, .1, .2], [1., .2, .2, .5])
# %jupyter_snippet 1
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [.08] * 3, [.2, .2, 1., .5])

#
# OPTIM 6D #########################################################
#


def cost(q):
    '''Compute score from a configuration'''
    M = robot.placement(q, 6)
    return norm(pin.log(M.inverse() * Mtarget).vector)


def callback(q):
    viz.applyConfiguration(boxID, Mtarget)
    viz.applyConfiguration(tipID, robot.placement(q, 6))
    viz.display(q)
    time.sleep(1e-2)


Mtarget = pin.SE3(pin.utils.rotate('x', 3.14 / 4), np.array([-0.5, 0.1, 0.2]))  # x,y,z
qopt = fmin_bfgs(cost, robot.q0, callback=callback)

print('The robot finally reached effector placement at\n', robot.placement(qopt, 6))
# %end_jupyter_snippet


class InvGeom6DTest(unittest.TestCase):
    def test_qopt_6d(self):
        Mopt = robot.placement(qopt, 6)
        self.assertTrue((np.abs(Mtarget.translation - Mopt.translation) < 1e-7).all())
        self.assertTrue((np.abs(Mtarget.translation - Mopt.translation) < 1e-7).all())
