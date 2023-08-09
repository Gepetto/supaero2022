import matplotlib.pylab as plt
import numpy as np


def plotOCSolutions(xs, us, pltAxes=None):
    """
    Plot xs and us.
    pltAxes should be a collection of two plt.axes typically produced by a call
    to plt.subplots.
    By default (none), a new figure is created.

    :param xs: state trajectory.
    :param us: control trajectory.
    :param pltAxes: a collection of 2 plt.axes produced by a call to plt.subplots,
    None by default.
    """
    if pltAxes is None:
        fig, pltAxes = plt.subplots(2, 1, figsize=(6.4, 8))

    # Plotting the state trajectories
    h = pltAxes[0].plot(xs)
    pltAxes[0].legend(h, [f"$x_{i+1}$" for i in range(len(xs[0]))])
    pltAxes[0].set_title("States")
    pltAxes[0].set_xlabel("time (nodes)")

    # Plotting the control trajectories
    h = pltAxes[1].plot(us)
    pltAxes[1].legend(h, [f"$u_{i+1}$" for i in range(len(us[0]))])
    pltAxes[1].set_title("Controls")
    pltAxes[1].set_xlabel("time (nodes)")


def plotConvergence(log, pltAxes=None):
    """
    Plot the content of the crocoddyl.CallbackLoger <log>.
    pltAxes should be a collection of two plt.axes typically produced by a
    call to plt.subplots.
    By default (none), a new figure is created.
    The function indeed calls the more detailed function below.

    :param log: crocoddyl.CallbackLoger
    :param pltAxes: a collection of 2 plt.axes produced by a call to plt.subplots,
    None by default.
    """
    if pltAxes is None:
        f, pltAxes = plt.subplots(5, 1, figsize=(6.4, 8))
    plotConvergenceDetailed(
        log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, pltAxes
    )


def plotConvergenceDetailed(costs, muLM, muV, gamma, theta, alpha, pltAxes):
    """
    Plot the cost, regularization <mulM> (in us) and <muV> (in xs), gradient
    norms <gamma>, stoping criterion <theta>, and line search step <alpha>.
    pltAxes must be a collection of 5 plt.axes typically produced by a call
    to plt.subplots.


    :param mulM: regularization in us
    :param mulV: regularization in xs
    :param gamma: gradient norm
    :param theta: stoping criterion <theta>
    :param alpha: line search step <alpha>
    :param pltAxes: collection of 5 plt.axes typically produced by a call
    to plt.subplots.
    """

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Plotting the total cost sequence
    pltAxes[0].set_ylabel("cost")
    pltAxes[0].plot(costs)

    # Ploting mu sequences
    pltAxes[1].set_ylabel("mu")
    pltAxes[1].plot(muLM, label="regularize u")
    pltAxes[1].plot(muV, label="regularize x")
    pltAxes[1].legend()

    # Plotting the gradient sequence (gamma and theta)
    pltAxes[2].set_ylabel("gamma")
    pltAxes[2].plot(gamma)
    pltAxes[2].legend(["gradient"])
    pltAxes[3].set_ylabel("theta")
    pltAxes[3].plot(theta)
    pltAxes[3].legend(["stopping crit"])

    # Plotting the alpha sequence
    pltAxes[4].set_ylabel("alpha")
    ind = np.arange(len(alpha))
    pltAxes[4].bar(ind, alpha)
    pltAxes[4].set_xlabel("iteration")


def displayTrajectory(viz, xs, dt=0.01, rate=-1):
    """Display a robot trajectory xs using Gepetto-viewer gui.

    :param robot: Robot wrapper
    :param xs: state trajectory
    :param dt: step duration
    :param rate: visualization rate
    """

    import time

    S = 1 if rate <= 0 else max(int(1 / dt / rate), 1)
    for i, x in enumerate(xs):
        if not i % S:
            viz.display(x[: viz.model.nq])
            time.sleep(dt * S)
    viz.display(xs[-1][: viz.model.nq])
