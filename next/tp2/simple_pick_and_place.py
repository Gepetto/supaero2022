import math
import time
import unittest

import numpy as np
import pinocchio as pin
import example_robot_data as robex

from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors

# %jupyter_snippet 1
robot = robex.load('ur5')
# %end_jupyter_snippet
NQ = robot.model.nq
NV = robot.model.nv

# Open the viewer
viz = MeshcatVisualizer(robot)

# %jupyter_snippet 2
# Add a red box in the viewer
ballID = "world/ball"
viz.addSphere(ballID, 0.1, colors.red)

# Place the ball at the position ( 0.5, 0.1, 0.2 )
# The viewer expect position and rotation, apppend the identity quaternion
q_ball = [0.5, 0.1, 0.2, 1, 0, 0, 0]
viz.applyConfiguration(ballID, q_ball)
# %end_jupyter_snippet

#
# PICK #############################################################
#

# Configuration for picking the box
# %jupyter_snippet 3
q0 = np.zeros(NQ)  # set the correct values here
q0[0] = -0.375
q0[1] = -1.2
q0[2] = 1.71
q0[3] = -q0[1] - q0[2]
q0[4] = q0[0]
q0[5] = 0.

viz.display(q0)
q = q0.copy()
# %end_jupyter_snippet
print("The robot is display with end effector on the red ball.")

#
# MOVE 3D #############################################################
#

print("Let's start the movement ...")

# %jupyter_snippet 4
# Random velocity of the robot driving the movement
vq = np.array([2., 0, 0, 4., 0, 0])

idx = robot.index('wrist_3_joint')
o_eff = robot.placement(q, idx).translation  # Position of end-eff wrt world at current configuration
o_ball = q_ball[:3]  # Position of ball wrt world
eff_ball = o_ball - o_eff  # Position of ball wrt eff

for i in range(50):
    # Chose new configuration of the robot
    q += vq / 40
    q[2] = 1.71 + math.sin(i * 0.05) / 2

    # Gets the new position of the ball
    o_ball = robot.placement(q, idx) * eff_ball

    # Display new configuration for robot and ball
    viz.applyConfiguration(ballID, o_ball.tolist() + [1, 0, 0, 0])
    viz.display(q)
    time.sleep(1e-2)
# %end_jupyter_snippet

#
# MOVE 6D #############################################################
#

q = q0.copy()
viz.display(q0)

# %jupyter_snippet 5
# Add a red box in the viewer
boxID = "world/box"
#viz.delete(ballID)
viz.addBox(boxID, [0.1, 0.2, 0.1], colors.magenta)

# Place the box at the position (0.5, 0.1, 0.2)
q_box = [0.5, 0.1, 0.2, 1, 0, 0, 0]
viz.applyConfiguration(boxID, q_box)
viz.applyConfiguration(ballID, [2,2,2,1,0,0,0])
# %end_jupyter_snippet

# %jupyter_snippet 6
q0 = np.zeros(NQ)
q0[0] = -0.375
q0[1] = -1.2
q0[2] = 1.71
q0[3] = -q0[1] - q0[2]
q0[4] = q0[0]

viz.display(q0)
q = q0.copy()
# %end_jupyter_snippet

print("Now moving with a 6D object ... ")

# %jupyter_snippet 7
# Random velocity of the robot driving the movement
vq = np.array([2., 0, 0, 4., 0, 0])

idx = robot.index('wrist_3_joint')
oMeff = robot.placement(q, idx)  # Placement of end-eff wrt world at current configuration
oMbox = pin.XYZQUATToSE3(q_box)  # Placement of box     wrt world
effMbox = oMeff.inverse() * oMbox  # Placement of box     wrt eff

for i in range(100):
    # Chose new configuration of the robot
    q += vq / 40
    q[2] = 1.71 + math.sin(i * 0.05) / 2

    # Gets the new position of the box
    oMbox = robot.placement(q, idx) * effMbox

    # Display new configuration for robot and box
    viz.applyConfiguration(boxID, oMbox)
    viz.display(q)
    time.sleep(1e-2)
# %end_jupyter_snippet

##########################################################################
### This last part is to automatically validate the versions of this example.
class SimplePickAndPlaceTest(unittest.TestCase):
    def test_oMbox_translation(self):
        self.assertTrue((np.abs(oMbox.translation - np.array([ 0.22085156, -0.6436716 ,  0.5632217 ])) < 1e-5).all())
