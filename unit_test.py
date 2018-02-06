import util as util
from Scene.Octree import BoundingVolume
from Shapes.sphere import Sphere
import numpy as np
from ray import Ray

########## unit tests

def testUtil() :

    # test isColor
    print("testing util.isColor")
    assertTrue(util.isColor([0, 0, 0]))
    assertFalse(util.isColor([0, 0]))
    assertFalse(util.isColor([0, 0, "0"]))
    assertFalse(util.isColor([0, 0, 0, 0]))
    assertFalse(util.isColor([0, 0, 0, 0, 123123123]))


    #test clip color
    print("testing utils.clipColor")
    assertEquals1DFloatArray(util.clipColor([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])
    assertEquals1DFloatArray(util.clipColor([1, 1, 1]), [1, 1, 1])

    assertEquals1DFloatArray(util.clipColor([255, 255, 255]), [1, 1 , 1])
    assertEquals1DFloatArray(util.clipColor([-255, -255, -1255]), [0, 0 , 0])

    assertEquals1DFloatArray(util.clipColor([0.5, 0.5, 0.5]), [0.5, 0.5 , 0.5])



    return

def testOctree():
    bv = BoundingVolume(single=True)
    bv.BBv2 = np.array([-1,-1,-1])
    bv.BBv8 = np.array([1,1,1])
    bv.finalizeAABB()

    s1 = Sphere([0,0,0],1,[0,0,0])

    print("testing BoundingVolume.isPointInBoundingVolume")
    assertTrue(bv.isPointInBoundingVolume(s1, [0,0,0]))
    assertTrue(bv.isPointInBoundingVolume(s1, [1,0,0]))
    assertTrue(bv.isPointInBoundingVolume(s1, [0,1,0]))
    assertTrue(bv.isPointInBoundingVolume(s1, [0,0,1]))
    assertTrue(bv.isPointInBoundingVolume(s1, [0,0.5,0]))
    assertTrue(bv.isPointInBoundingVolume(s1, [0,1,1]))


    assertFalse(bv.isPointInBoundingVolume(s1, [0,0,1.1]))
    assertFalse(bv.isPointInBoundingVolume(s1, [0,1,1.1]))
    assertFalse(bv.isPointInBoundingVolume(s1, [0,1,-1.1]))
    assertFalse(bv.isPointInBoundingVolume(s1, [9999,0,-999]))
    assertFalse(bv.isPointInBoundingVolume(s1, [9999,0,-999]))

    print("testing BoundingVolume.intersectsWithRay axis aligned")
    rFromLeftMid = Ray(np.array([0.0,-10.0,0.0]), np.array([0.0,0.9, 0.0]))
    rFromRightMid = Ray(np.array([0,10,0]), np.array([0,-0.8, 0]))

    fromTopMid = Ray(np.array([-10,0,0]), np.array([1,0, 0]))
    fromBottomMid = Ray(np.array([10,0,0]), np.array([-0.9,0, 0]))

    fromFrontMid = Ray(np.array([0,0,-10]), np.array([0,0, 1]))
    fromBackMid = Ray(np.array([0,0,10]), np.array([0,0, -0.7]))

    assertTrue(bv.intersectsWithRay(rFromLeftMid))
    assertTrue(bv.intersectsWithRay(rFromRightMid))
    assertTrue(bv.intersectsWithRay(fromTopMid))
    assertTrue(bv.intersectsWithRay(fromBottomMid))
    assertTrue(bv.intersectsWithRay(fromFrontMid))
    assertTrue(bv.intersectsWithRay(fromBackMid))

    rFromLeftMid.o += np.array([10,0,0])
    assertFalse(bv.intersectsWithRay(rFromLeftMid))
    #todo add some more negative tests


    return



########## test utils
# functions should be designed in a way so that when an
# assertion fails, an exception should be raised

def assertTrue(expr) :
    if not expr :
        raise Exception()
    return

def assertFalse(expr) :
    if expr :
        raise Exception()
    return

def assertEquals(val1, val2) :
    if val1 != val2 :
        raise Exception()
    return

def assertNotEquals(val1, val2) :
    if val1 == val2 :
        raise Exception()
    return

def assertEquals1DFloatArray(val1, val2) :
    if len(val1) != len(val2) :
        raise Exception()

    notEqual = False
    for n in range(0, len(val1)) :
        notEqual |= abs(val1[n] - val2[n]) > 0.00001
        if notEqual :
            break

    if notEqual :
        raise Exception()
    return


def runTests() :
    testUtil()
    testOctree()

    return




runTests()