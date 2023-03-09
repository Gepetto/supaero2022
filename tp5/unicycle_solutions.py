import crocoddyl
from unicycle_utils import plotUnicycleSolution
from croco_utils import plotOCSolutions,plotConvergence
import numpy as np
import matplotlib.pylab as plt
import unittest

# %jupyter_snippet hyperparams
### HYPER PARAMS: horizon and initial state
T  = 100
x0 = np.array([-1,-1,1])
# %end_jupyter_snippet

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
plotOCSolutions(log.xs, log.us)
plotConvergence(log)
plotUnicycleSolution(log.xs)

print('Type plt.show() to display the result.')

### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class UnicycleTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        self.assertTrue( len(ddp.xs) == len(ddp.us)+1 )
        self.assertTrue( np.allclose(ddp.xs[0],ddp.problem.x0) )
        self.assertTrue( ddp.stop<1e-6 )
        
if __name__ == "__main__":
    UnicycleTest().test_logs()
