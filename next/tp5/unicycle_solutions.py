import crocoddyl
from unicycle_utils import plotUnicycleSolution
import numpy as np
import matplotlib.pylab as plt; plt.ion()

### HYPER PARAMS: horizon and initial state
T  = 100
x0 = np.array([-1,-1,1])

### PROBLEM DEFINITION

model = crocoddyl.ActionModelUnicycle()
# %jupyter_snippet termmodel
model_term = crocoddyl.ActionModelUnicycle()

model_term.costWeights = np.matrix([
    100,   # state weight
    0  # control weight
]).T

# Define integral+terminal models
problem = crocoddyl.ShootingProblem(x0, [ model ] * T, model_term)
ddp = crocoddyl.SolverDDP(problem)
# %end_jupyter_snippet

# Add solvers for verbosity and plots
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

### SOLVER PROBLEM
done = ddp.solve()
assert(done)

### PLOT 
log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2, show=False)
plotUnicycleSolution(log.xs, figIndex=3, show=True)

