import time

import numpy as np


def change_box_pose_in_viz(viz, quat):
    p0 = np.random.rand(3)
    p1 = np.random.rand(3)

    for t in np.arange(0, 1, 0.01):
        p = p0 * (1 - t) + p1 * t
        viz.applyConfiguration("world/box", list(p) + list(quat.coeffs()))
        time.sleep(0.01)
