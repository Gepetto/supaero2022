import pinocchio as pin
import numpy as np
import matplotlib.pylab as plt; plt.ion()
from numpy.linalg import inv, pinv, norm
import time
from tp6.robot_hand import RobotHand
from tp6.meshcat_viewer_wrapper import MeshcatVisualizer

robot = RobotHand()
viz = MeshcatVisualizer(robot,url='tcp://127.0.0.1:6002')#classical')

q = robot.q0.copy()
vq = np.zeros(robot.model.nv)

Kp = 50.
Kv = 2 * np.sqrt(Kp)
dt = 1e-3

from tp6.traj_ref import TrajRef
qdes = TrajRef(robot.q0,omega = np.array([0,.1,1,1.5,2.5,-1,-1.5,-2.5,.1,.2,.3,.4,.5,.6]),amplitude=1.5)

hq    = []   ### For storing the logs of measured trajectory q
hqdes = []   ### For storing the logs of desired trajectory qdes
for i in range(10000):
    t = i*dt
    
    M = pin.crba(robot.model, robot.data, q)
    b = pin.rnea(robot.model, robot.data, q, vq, np.zeros(robot.model.nv))

    tauq = -Kp*(q-qdes(t)) - Kv*(vq-qdes.velocity(t)) + qdes.acceleration(t)

    aq = inv(M) @ (tauq - b)
    vq += aq * dt
    q = pin.integrate(robot.model, q, vq * dt)

    TDISP = 50e-3    # Display every 50ms
    if not i % int(TDISP/dt):  # Only display once in a while ...
        viz.display(q)
        time.sleep(TDISP)

    hq.append(q.copy())
    hqdes.append(qdes.copy())

