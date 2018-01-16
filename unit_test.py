import util as util

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

    return




runTests()