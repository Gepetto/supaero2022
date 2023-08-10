from time import sleep

import numpy as np
from numpy.linalg import norm
from pinocchio import SE3, AngleAxis, Quaternion

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

viz = MeshcatVisualizer()


def _lerp(p0, p1, t):
    return (1 - t) * p0 + t * p1


def slerp(q0, q1, t):
    assert t >= 0 and t <= 1
    a = AngleAxis(q0.inverse() * q1)
    return Quaternion(AngleAxis(a.angle * t, a.axis))


def nlerp(q0, q1, t):
    q0 = q0.coeffs()
    q1 = q1.coeffs()
    lerp = _lerp(q0, q1, t)
    lerp /= norm(lerp)
    return Quaternion(lerp[3], *list(lerp[:3]))


q0 = Quaternion(SE3.Random().rotation)
q1 = Quaternion(SE3.Random().rotation)
viz.applyConfiguration("world/box", [0, 0, 0] + list(q0.coeffs()))
sleep(0.1)
for t in np.arange(0, 1, 0.01):
    q = nlerp(q0, q1, t)
    viz.applyConfiguration("world/box", [0, 0, 0] + list(q.coeffs()))
    sleep(0.01)
sleep(0.1)
viz.applyConfiguration("world/box", [0, 0, 0] + list(q1.coeffs()))
