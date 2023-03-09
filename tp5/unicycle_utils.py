import matplotlib.pyplot as plt
import numpy as np


def plotUnicycle(x,pltAx):
    sc, delta = .1, .1
    a, b, th = x
    c, s = np.cos(th), np.sin(th)
    refs = [
        pltAx.arrow(a - sc / 2 * c - delta * s, b - sc / 2 * s + delta * c, c * sc, s * sc, head_width=.05),
        pltAx.arrow(a - sc / 2 * c + delta * s, b - sc / 2 * s - delta * c, c * sc, s * sc, head_width=.05)
    ]
    return refs


def plotUnicycleSolution(xs, pltAx=None):
    if pltAx is None:
        f,pltAx = plt.subplots(1,1, figsize=(6.4, 6.4))
    for x in xs:
        plotUnicycle(x,pltAx)
    pltAx.axis([-2, 2., -2., 2.])
