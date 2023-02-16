'''
Inverse kinematics (close loop / iterative) for a mobile manipulator.
This second example is built upon the more simple tp3/inverse_kinematics.py.
It adds a frame corresponding to the robot gaze, and provides a multi-objective control loop
where the robot servo both its gripper and its gaze in two hierachized tasks.
'''

import pinocchio as pin
import numpy as np
import time
from numpy.linalg import pinv,inv,norm,svd,eig
from tp3.tiago_loader import loadTiago
import matplotlib.pylab as plt; plt.ion()
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
import unittest

# %jupyter_snippet robot
robot = loadTiago(addGazeFrame=True)
viz = MeshcatVisualizer(robot)
# %end_jupyter_snippet

NQ = robot.model.nq
NV = robot.model.nv
IDX_TOOL = robot.model.getFrameId('frametool')
IDX_BASIS = robot.model.getFrameId('framebasis')
# %jupyter_snippet gaze
IDX_GAZE = robot.model.getFrameId('framegaze')

# Add a small ball as a visual target to be reached by the robot
ball = np.array([ 1.2,0.5,1.1 ])
viz.addSphere('ball', .05, [ .8,.1,.5, .8] )
viz.applyConfiguration('ball', list(ball)+[0,0,0,1])
# %end_jupyter_snippet

# Goal placement, and integration in the viewer of the goal.
oMgoal = pin.SE3(pin.Quaternion(-0.5, 0.58, -0.39, 0.52).normalized().matrix(),
                np.array([1.2, .4, .7]))
viz.addBox('goal', [.1,.1,.1], [ .1,.1,.5, .6] )
viz.applyConfiguration('goal',oMgoal)

# Integration step.
DT = 1e-2

# Robot initial configuration.
q0 = np.array([ 0.  ,  0.  ,  0.  ,  1.  ,  0.18,  1.37, -0.24, -0.98,  0.98,
                0.  ,  0.  ,  0.  ,  0.  , -0.13,  0.  ,  0.  ,  0.  ,  0.  ])

# %jupyter_snippet gaze_loop
q = q0.copy()
herr = [] # Log the value of the error between gaze and ball.
# Loop on an inverse kinematics for 200 iterations.
for i in range(200):  # Integrate over 2 second of robot life

    # Run the algorithms that outputs values in robot.data
    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.computeJointJacobians(robot.model,robot.data,q)

    # Placement from world frame o to frame f oMgaze
    oMgaze = robot.data.oMf[IDX_GAZE]

    # 6D jacobian in local frame
    o_Jgaze3 = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_GAZE,pin.LOCAL_WORLD_ALIGNED)[:3,:]

    # vector from gaze to ball, in world frame
    o_GazeBall = oMgaze.translation-ball
    
    vq = -pinv(o_Jgaze3) @ o_GazeBall

    q = pin.integrate(robot.model,q, vq * DT)
    viz.display(q)
    time.sleep(1e-3)

    herr.append(o_GazeBall) 
# %end_jupyter_snippet

# %jupyter_snippet multi
q = q0.copy()
herr = [] # Log the value of the error between tool and goal.
herr2 = [] # Log the value of the error between gaze and ball.
# Loop on an inverse kinematics for 200 iterations.
for i in range(200):  # Integrate over 2 second of robot life

    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.computeJointJacobians(robot.model,robot.data,q)

    # Gaze task
    oMgaze = robot.data.oMf[IDX_GAZE]
    o_Jgaze3 = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_GAZE,pin.LOCAL_WORLD_ALIGNED)[:3,:]
    o_GazeBall = oMgaze.translation-ball

    # Tool task
    oMtool = robot.data.oMf[IDX_TOOL]
    o_Jtool3 = pin.computeFrameJacobian(robot.model,robot.data,q,IDX_TOOL,pin.LOCAL_WORLD_ALIGNED)[:3,:]
    o_TG = oMtool.translation-oMgoal.translation
    
    vq = -pinv(o_Jtool3) @ o_TG
    Ptool = np.eye(robot.nv)-pinv(o_Jtool3) @ o_Jtool3
    vq += pinv(o_Jgaze3 @ Ptool) @ (-o_GazeBall - o_Jgaze3 @ vq)

    q = pin.integrate(robot.model,q, vq * DT)
    viz.display(q)
    time.sleep(1e-3)

    herr.append(o_TG)
    herr2.append(o_GazeBall) 
# %end_jupyter_snippet


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class ControlHeadTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        self.assertTrue(norm(herr[0])/5>norm(herr[-1])) # Check error decrease
        self.assertTrue(norm(herr2[0])/5>norm(herr2[-1])) # Check error decrease

if __name__ == "__main__":
    ControlHeadTest().test_logs()

