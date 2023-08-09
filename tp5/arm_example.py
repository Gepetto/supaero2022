"""
# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside 
  DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.
"""

import crocoddyl
import example_robot_data as robex
import numpy as np
import pinocchio as pin
from croco_utils import displayTrajectory, plotConvergence, plotOCSolutions

# %jupyter_snippet robexload
# First, let's load the Pinocchio model for the Talos arm.
robot = robex.load("talos_arm")
# %end_jupyter_snippet

# %jupyter_snippet robot_model
# Set robot model
robot_model = robot.model
robot_model.armature = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]) * 5
robot_model.q0 = np.array([3.5, 2, 2, 0, 0, 0, 0])
robot_model.x0 = np.concatenate([robot_model.q0, np.zeros(robot_model.nv)])
robot_model.gravity *= 0
# %end_jupyter_snippet

# Configure tasks
# %jupyter_snippet taskid
FRAME_TIP = robot_model.getFrameId("gripper_left_fingertip_3_link")
goal = np.array([0.2, 0.5, 0.5])
# %end_jupyter_snippet

# Configure viewer
# %jupyter_snippet viz
from utils.meshcat_viewer_wrapper import MeshcatVisualizer  # noqa E402

viz = MeshcatVisualizer(robot)
viz.display(robot_model.q0)
viz.addBox("world/box", [0.1, 0.1, 0.1], [1.0, 0, 0, 1])
viz.addBox("world/goal", [0.1, 0.1, 0.1], [0, 1, 0, 1])
viz.applyConfiguration("world/goal", [0.2, 0.5, 0.5, 0, 0, 0, 1])
# %end_jupyter_snippet

# Create a cost model per the running and terminal action model.
# %jupyter_snippet stateandcosts
state = crocoddyl.StateMultibody(robot_model)
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
# %end_jupyter_snippet

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.

# %jupyter_snippet costs
# Cost for 3d tracking || p(q) - pref ||**2
goalTrackingRes = crocoddyl.ResidualModelFrameTranslation(state, FRAME_TIP, goal)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingRes)

# Cost for 6d tracking  || log( M(q)^-1 Mref ) ||**2
Mref = pin.SE3(pin.utils.rpyToMatrix(0, np.pi / 2, -np.pi / 2), goal)
goal6TrackingRes = crocoddyl.ResidualModelFramePlacement(state, FRAME_TIP, Mref)
goal6TrackingCost = crocoddyl.CostModelResidual(state, goal6TrackingRes)

# Cost for state regularization || x - x* ||**2
xRegWeights = crocoddyl.ActivationModelWeightedQuad(
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2.0])
)
xRegRes = crocoddyl.ResidualModelState(state, robot_model.x0)
xRegCost = crocoddyl.CostModelResidual(state, xRegWeights, xRegRes)

# Cost for control regularization || u - g(q) ||**2
uRegRes = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uRegRes)

# Terminal cost for state regularization || x - x* ||**2
xRegWeightsT = crocoddyl.ActivationModelWeightedQuad(
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2.0])
)
xRegResT = crocoddyl.ResidualModelState(state, robot_model.x0)
xRegCostT = crocoddyl.CostModelResidual(state, xRegWeightsT, xRegResT)
# %end_jupyter_snippet

# Then let's added the running and terminal cost functions
# %jupyter_snippet addcosts
runningCostModel.addCost("gripperPose", goalTrackingCost, 0.001)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-6)
terminalCostModel.addCost("gripperPose", goal6TrackingCost, 10)
terminalCostModel.addCost("xReg", xRegCostT, 0.01)
# %end_jupyter_snippet

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.
# %jupyter_snippet iam
actuationModel = crocoddyl.ActuationModelFull(state)
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, runningCostModel
    ),
    dt,
)
runningModel.differential.armature = robot_model.armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, terminalCostModel
    ),
    0.0,
)
terminalModel.differential.armature = robot_model.armature
# %end_jupyter_snippet

# For this optimal control problem, we define 100 knots (or running action
# models) plus a terminal knot
# %jupyter_snippet shoot
T = 100
problem = crocoddyl.ShootingProblem(robot_model.x0, [runningModel] * T, terminalModel)
# %end_jupyter_snippet

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
# %jupyter_snippet callbacks
ddp.setCallbacks(
    [
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
    ]
)
# %end_jupyter_snippet

# Solving it with the DDP algorithm
ddp.solve([], [], 1000)  # xs_init,us_init,maxiter
assert ddp.stop == 1.9384159634520916e-10

# Plotting the solution and the DDP convergence
plotOCSolutions(ddp.xs, ddp.us)
plotConvergence(ddp.getCallbacks()[0])
print("Type plt.show() to display the plots")

# Visualizing the solution in gepetto-viewer
displayTrajectory(viz, ddp.xs, ddp.problem.runningModels[0].dt, 12)
