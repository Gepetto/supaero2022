'''
Tiago loader accounting for the first planar joint giving robot mobility.
'''

import numpy as np
import pinocchio as pin
import example_robot_data as robex
import hppfcl
from os.path import dirname, exists, join
import unittest

class TiagoLoader(object):
    #path = ''
    #urdf_filename = ''
    srdf_filename = ''
    urdf_subpath = 'robots'
    srdf_subpath = 'srdf'
    ref_posture = 'half_sitting'
    has_rotor_parameters = False
    free_flyer = True
    verbose = False
    path = "tiago_description"
    urdf_filename = "tiago_no_hand.urdf"

    def __init__(self):
        urdf_path = join(self.path, self.urdf_subpath, self.urdf_filename)
        self.model_path = robex.getModelPath(urdf_path, self.verbose)
        self.urdf_path = join(self.model_path, urdf_path)
        self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, [join(self.model_path, '../..')],
                                                    pin.JointModelPlanar() if self.free_flyer else None)

        if self.srdf_filename:
            self.srdf_path = join(self.model_path, self.path, self.srdf_subpath, self.srdf_filename)
            self.q0 = readParamsFromSrdf(self.robot.model, self.srdf_path, self.verbose, self.has_rotor_parameters,
                                         self.ref_posture)
        else:
            self.srdf_path = None
            self.q0 = None

        if self.free_flyer:
            self.addFreeFlyerJointLimits()

    def addFreeFlyerJointLimits(self):
        ub = self.robot.model.upperPositionLimit
        ub[:self.robot.model.joints[1].nq] = 1
        self.robot.model.upperPositionLimit = ub
        lb = self.robot.model.lowerPositionLimit
        lb[:self.robot.model.joints[1].nq] = -1
        self.robot.model.lowerPositionLimit = lb


def loadTiago(addGazeFrame=False):
    '''
    Load a tiago model, without the hand, and with the two following modifications wrt example_robot_data.
    - first, the first joint is a planar (x,y,cos,sin) joint, while it is a fixed robot in example robot data.
    - second, two visual models of a frame have been added to two new op-frame, "tool0" on the robot hand, and "basis0" in 
    front of the basis.
    '''

    
    robot = TiagoLoader().robot
    geom = robot.visual_model

    X = pin.utils.rotate('y', np.pi/2)
    Y = pin.utils.rotate('x',-np.pi/2)
    Z = np.eye(3)
    
    L = .3
    cyl=hppfcl.Cylinder(L/30,L)
    med = np.array([0,0,L/2])
    
    # ---------------------------------------------------------------------------
    # Add a frame visualisation in the effector.
    
    FIDX = robot.model.getFrameId('wrist_ft_tool_link')
    JIDX = robot.model.frames[FIDX].parent
    
    eff = np.array([0,0,.08])
    FIDX = robot.model.addFrame(pin.Frame('frametool',JIDX,FIDX,pin.SE3(Z,eff),pin.FrameType.OP_FRAME))
    
    geom.addGeometryObject(pin.GeometryObject('axis_x',FIDX,JIDX,cyl,pin.SE3(X,X@med+eff)))
    geom.geometryObjects[-1].meshColor = np.array([1,0,0,1.])
    
    geom.addGeometryObject(pin.GeometryObject('axis_y',FIDX,JIDX,cyl,pin.SE3(Y,Y@med+eff)))
    geom.geometryObjects[-1].meshColor = np.array([0,1,0,1.])
    
    geom.addGeometryObject(pin.GeometryObject('axis_z',FIDX,JIDX,cyl,pin.SE3(Z,Z@med+eff)))
    geom.geometryObjects[-1].meshColor = np.array([0,0,1,1.])
    
    # ---------------------------------------------------------------------------
    # Add a frame visualisation in front of the basis.
    
    FIDX = robot.model.getFrameId('base_link')
    JIDX = robot.model.frames[FIDX].parent
    
    eff = np.array([.3,0,.15])
    FIDX = robot.model.addFrame(pin.Frame('framebasis',JIDX,FIDX,pin.SE3(Z,eff),pin.FrameType.OP_FRAME))
    
    geom.addGeometryObject(pin.GeometryObject('axis2_x',FIDX,JIDX,cyl,pin.SE3(X,X@med+eff)))
    geom.geometryObjects[-1].meshColor = np.array([1,0,0,1.])
    
    geom.addGeometryObject(pin.GeometryObject('axi2_y',FIDX,JIDX,cyl,pin.SE3(Y,Y@med+eff)))
    geom.geometryObjects[-1].meshColor = np.array([0,1,0,1.])
    
    geom.addGeometryObject(pin.GeometryObject('axis2_z',FIDX,JIDX,cyl,pin.SE3(Z,Z@med+eff)))
    geom.geometryObjects[-1].meshColor = np.array([0,0,1,1.])

    # ---------------------------------------------------------------------------
    # Add a frame visualisation in front of the head.

    if addGazeFrame:
        L = .05
        cyl=hppfcl.Cylinder(L/30,L)
        med = np.array([0,0,L/2])

        FIDX = robot.model.getFrameId('xtion_joint')
        JIDX = robot.model.frames[FIDX].parent
        
        eff = np.array([0.4,0.0,0.0])
        FIDX = robot.model.addFrame(pin.Frame('framegaze',JIDX,FIDX,pin.SE3(Z,eff),pin.FrameType.OP_FRAME))
        
        geom.addGeometryObject(pin.GeometryObject('axisgaze_x',FIDX,JIDX,cyl,pin.SE3(X,X@med+eff)))
        geom.geometryObjects[-1].meshColor = np.array([1,0,0,1.])
        
        geom.addGeometryObject(pin.GeometryObject('axisgaze_y',FIDX,JIDX,cyl,pin.SE3(Y,Y@med+eff)))
        geom.geometryObjects[-1].meshColor = np.array([0,1,0,1.])
    
        geom.addGeometryObject(pin.GeometryObject('axisgaze_z',FIDX,JIDX,cyl,pin.SE3(Z,Z@med+eff)))
        geom.geometryObjects[-1].meshColor = np.array([0,0,1,1.])

    # -------------------------------------------------------------------------------
    # Regenerate the data from the new models.
    
    robot.q0 = np.array([1,1,1,0]+[0]*(robot.model.nq-4))

    robot.data = robot.model.createData()
    robot.visual_data = robot.visual_model.createData()
    
    return robot

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class TiagoTest(unittest.TestCase):
    def test_load(self):
        print(self.__class__.__name__)
        robot = loadTiago()
        self.assertTrue(robot.model.nq==18)
        self.assertTrue(robot.model.nv==15)

if __name__ == "__main__":
    from utils.meshcat_viewer_wrapper import MeshcatVisualizer
    robot = loadTiago()
    viz = MeshcatVisualizer(robot)
    viz.display(robot.q0)
    TiagoTest().test_load()



