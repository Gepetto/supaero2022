'''
Stand-alone program to optimize the placement of a 2d robot, where the decision variables
are the placement of the 3 bodies of the robot. BFGS and SLSQP solvers are used.
'''

import time
import numpy as np
from scipy.optimize import fmin_bfgs,fmin_slsqp
from utils.meshcat_viewer_wrapper import MeshcatVisualizer,translation2d,planar
from numpy.linalg import norm,inv,pinv,svd,eig

viz = MeshcatVisualizer(url='classical')

viz.addSphere('joint1',.1,[1,0,0,1])
viz.addSphere('joint2',.1,[1,0,0,1])
viz.addSphere('joint3',.1,[1,0,0,1])
viz.addCylinder('arm1',.75,.05,[.65,.65,.65,1])
viz.addCylinder('arm2',.75,.05,[.65,.65,.65,1])
viz.addSphere('target',.1001,[0,.8,.1,1])

# %jupyter_snippet display
def display_9(ps):
    '''Display the robot in the Viewer. '''
    assert (ps.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps
    viz.applyConfiguration('joint1',planar(x1,                  y1,                  t1))
    viz.applyConfiguration('arm1'  ,planar(x1 + np.cos(t1) / 2, x1 + np.sin(t1) / 2, t1))
    viz.applyConfiguration('joint2',planar(x2,                  y2,                  t2))
    viz.applyConfiguration('arm2'  ,planar(x2 + np.cos(t2) / 2, y2 + np.sin(t2) / 2, t2))
    viz.applyConfiguration('joint3',planar(x3,                  y3,                  t3))
# %end_jupyter_snippet

# %jupyter_snippet endeffector
def endeffector_9(ps):
    assert (ps.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps
    return np.array([x3, y3])
# %end_jupyter_snippet

target = np.array([.5, .5])
viz.applyConfiguration('target',translation2d(target[0],target[1]))

# %jupyter_snippet cost
def cost_9(ps):
    eff = endeffector_9(ps)
    return norm(eff - target)**2
# %end_jupyter_snippet

# %jupyter_snippet constraint
def constraint_9(ps):
    assert (ps.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps
    res = np.zeros(6)
    res[0] = x1 - 0
    res[1] = y1 - 0
    res[2] = x1 + np.cos(t1) - x2
    res[3] = y1 + np.sin(t1) - y2
    res[4] = x2 + np.cos(t2) - x3
    res[5] = y2 + np.sin(t2) - y3
    return res
# %end_jupyter_snippet

# %jupyter_snippet penalty
def penalty(ps):
    return cost_9(ps) + 10 * sum(np.square(constraint_9(ps)))
# %end_jupyter_snippet

# %jupyter_snippet callback
def callback_9(ps):
    display_9(ps)
    time.sleep(.5)
# %end_jupyter_snippet

x0 = np.array([ 0.0,] * 9)

with_bfgs = 0
if with_bfgs:
    xopt = fmin_bfgs(penalty, x0, callback=callback_9)
else:
    xopt = fmin_slsqp(cost_9, x0, callback=callback_9, f_eqcons=constraint_9, iprint=2, full_output=1)[0]
print('\n *** Xopt = %s\n\n\n\n' % xopt)
