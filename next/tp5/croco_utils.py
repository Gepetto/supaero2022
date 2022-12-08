def displayTrajectory(viz, xs, dt=0.01, rate=-1):
    """  Display a robot trajectory xs using Gepetto-viewer gui.

    :param robot: Robot wrapper
    :param xs: state trajectory
    :param dt: step duration
    :param rate: visualization rate
    """

    import time
    S = 1 if rate <= 0 else max(int(1/dt/rate), 1)
    for i, x in enumerate(xs):
        if not i % S:
            viz.display(x[:viz.model.nq])
            time.sleep(dt*S)
    viz.display(xs[-1][:viz.model.nq])
    
