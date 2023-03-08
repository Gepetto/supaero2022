'''
# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.
'''

import sys
WITHDISPLAY = 'display' in sys.argv
WITHPLOT = 'plot' in sys.argv

import crocoddyl; crocoddyl.switchToNumpyMatrix()
import pinocchio
import numpy as np
import example_robot_data

# First, let's load the Pinocchio model for the Talos arm.
robot = example_robot_data.load('talos_arm')

# Set robot model
robot_model = robot.model
robot_model.armature =np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.])*5
robot_model.q0 = np.array([3.5,2,2,0,0,0,0])
robot_model.x0 = np.concatenate([robot_model.q0, pinocchio.utils.zero(robot_model.nv)])
robot_model.gravity *= 0

# Configure task
FRAME_TIP = robot_model.getFrameId("gripper_left_fingertip_3_link")
goal = np.array([.2,0.5,.5])

# Configure viewer
from tp3.meshcat_viewer_wrapper import MeshcatVisualizer
viz = MeshcatVisualizer(robot,url='tcp://127.0.0.1:6004')#classical')
viz.display(robot_model.q0)
viz.addBox('world/box',[.1,.1,.1], [1.,0,0,1])
viz.addBox('world/goal',[.1,.1,.1],[0,1,0,1])
viz.applyConfiguration('world/goal',[0.2,0.5,.5,0,0,0,1])

# Create a cost model per the running and terminal action model.
state = crocoddyl.StateMultibody(robot_model)
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
Mref = crocoddyl.FramePlacement(FRAME_TIP,pinocchio.SE3(np.eye(3), goal))
#goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref)
pref = crocoddyl.FrameTranslation(FRAME_TIP,goal)
goalTrackingCost = crocoddyl.CostModelFrameTranslation(state, pref)
weights=crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1, 1,1,1,1,2,2,2.]))
xRegCost = crocoddyl.CostModelState(state,weights,robot_model.x0)
uRegCost = crocoddyl.CostModelControl(state)
weightsT=crocoddyl.ActivationModelWeightedQuad(np.array([.01,.01,.01,.01,.01,.01,.01, 1,1,1,1,2,2,2.]))
xRegCostT = crocoddyl.CostModelState(state,weights,robot_model.x0)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, .001)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-6)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 10)
terminalCostModel.addCost("xReg", xRegCostT, .01)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.
actuationModel = crocoddyl.ActuationModelFull(state)
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModel), dt)
runningModel.differential.armature = robot_model.armature
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel), 0.)
terminalModel.differential.armature = robot_model.armature

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 100
problem = crocoddyl.ShootingProblem(robot_model.x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose(),
])

# Solving it with the DDP algorithm
ddp.solve([],[],1000)  # xs_init,us_init,maxiter

# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = ddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    import tp7.croco_utils as crocutils
    crocutils.displayTrajectory(viz,ddp.xs,ddp.problem.runningModels[0].dt,12)
