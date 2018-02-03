import numpy as np
import time
import copy

"""
Bounding Box and Bounding Volume are the same in the context of this file
"""

# todo fix v2 / v8 confusion

class Octree():
    """
    Octree will divide scene recursively and fit objects into boxes efficiently
    List of objects that intersect with ray in scene can be obtained with intersect(...)
    """

    # How large the octree will span in the world (in each axis, from coordinate origin)
    # Everying outside will cause undefined behavior (most likely won't be picked up by intersect() )
    OCTREE_COORDINATE_MAX = 1000

    # Maximum divisions
    MAX_DEPTH = 10

    def __init__(self, shapeArray):
        """
        Creates octree from given array of shapes. See create() for more information
        :param shapeArray: [] of Shapes.shape derivates
        """
        self.rootBoundingVolume = BoundingVolume(single=True)
        self.rootBoundingVolume.BBv2 = np.array([-Octree.OCTREE_COORDINATE_MAX,
                                                -Octree.OCTREE_COORDINATE_MAX,
                                                -Octree.OCTREE_COORDINATE_MAX])
        self.rootBoundingVolume.BBv8 = np.array([Octree.OCTREE_COORDINATE_MAX,
                                                Octree.OCTREE_COORDINATE_MAX,
                                                Octree.OCTREE_COORDINATE_MAX])
        self.rootBoundingVolume.finalizeAABB()

        self.rootNode = 0
        self.biggestShapeListOnLeaf = 0
        self.leafCount = 0
        self.leafDepthSum = 0

        self.create(shapeArray)
        return


    def create(self, shapeArray):
        """
        Calculates octree from given array of shapes. Call again to recreate.
        :param shapeArray: [] of Shapes.shape derivates
        :return:
        """
        t0 = time.process_time()

        # octree is only valid for object set given on creation
        # to avoid errors we copy the list
        # todo maybe do deep copy if strange errors arise
        self.rootNode = OctreeNode(self.rootBoundingVolume, copy.copy(shapeArray), 0)
        self.rootNode.initializeOctants()

        val =  self._create(self.rootNode, 1)

        print("Octree created in %2.1f" % ((time.process_time() - t0)*1000),
              " ms   Biggest shapeList: ", self.biggestShapeListOnLeaf,
              " Leaf Count: ", self.leafCount,
              " Average Leaf Depth: %3.1f" % (self.leafDepthSum / self.leafCount))
        return val

    def _create(self, node, recursionLevel):
        if recursionLevel > Octree.MAX_DEPTH:
            self.logBiggestLeafShapeList(node)
            return

        # initialize all octants of this node
        for n in range(8):
            node.octants[n].initializeOctants()


        # kill all newly intialized octants when every one of them has the same shapeList as this node
        # this means that this node only contains very big objects that fill the nodes bounding volume completely
        allOctantsSameAsNode = True
        for n in range(8):
            # if lengths are not the same, no point in checking further
            # one octant is different than the node
            if len(node.shapeList) == len(node.octants[n].shapeList):
                thisOctantSameAsNode = True
                for i in range(len(node.shapeList)):
                    if not node.shapeList[i] is node.octants[n].shapeList[i]:
                        thisOctantSameAsNode = False
                        break
                if not thisOctantSameAsNode:
                    allOctantsSameAsNode = False
                    break
            else:
                allOctantsSameAsNode = False
                break

        if allOctantsSameAsNode:
            #print("Pruned a complete Node, Reason: All octants are same as parent node")
            node.uninitialize()
            self.logBiggestLeafShapeList(node)
            return

        # prune all newly intialized octants that contribute nothing to
        # faster lookup within the octree

        # check sizes of octant's shapeLists
        countOctantsWithShapes = 0
        for n in range(8):
            # prune empty octants
            if len(node.octants[n].shapeList) == 0:
                node.octants[n].uninitialize()
                #print("Pruned Octant, Reason: Empty shapeList")
            else:
                countOctantsWithShapes += 1


        # prune complete node if all nodes are empty
        if countOctantsWithShapes == 0:
            #print("Pruned a complete Node, Reason: All octants are empty")
            node.uninitialize()
            self.logBiggestLeafShapeList(node)
            return

        # prune this node if it has only one octant with shapes
        #if countOctantsWithShapes == 1:
        #    #print("Pruned a complete Node, Reason: All but one octant empty")
        #    node.uninitialize()
        #    self.logBiggestLeafShapeList(node)
        #    return

        # rest of octants are fine, divide that space further
        for n in range(8):
            if node.octants[n].isInitialized():
                self._create(node.octants[n], recursionLevel + 1)
        return

    def logBiggestLeafShapeList(self, node):
        # log biggest shape list
        self.biggestShapeListOnLeaf = max(self.biggestShapeListOnLeaf, len(node.shapeList))

        # find out node depth
        # root node has parent 0
        depth = -1
        currNode = node
        while(isinstance(currNode, OctreeNode)):
            depth += 1
            currNode = currNode.parentNode

        self.leafCount += 1
        self.leafDepthSum += depth

        return

    def intersect(self, ray):




        return False

class OctreeNode():
    """
    Represents one octree node. Divides given bounding volume into 8 smaller pieces (octants)
    Contains:
        - 8 Octants
        - List of shapes that intersect with 'bounding volume'
    To initialize octants call initializeOctants()

    Octants are enumerated as follows (figure is aligned with coordinate system in camera class)
        vN.... coordinates of bounding volume

        front:
        v2---+----v3
        | 0  | 3  |
        |    |    |
        |----+----|
        | 1  | 2  |
        |    |    |
        v6---+----v7


        back:
        v1---+----v4
        | 4  | 7  |
        |    |    |
        |----+----|
        | 5  | 6  |
        |    |    |
        v5---+----v8

        0... front-top-left
        1... front-bottom-left
        2... front-bottom-right
        3... front-top-right

        4... back-top-left
        5... back-bottom-left
        6... back-bottom-right
        7... back-top-right
    """
    def __init__(self, boundingVolume, shapeList, parentNode):
        """
        Initializes node.
        Creates bounding volumes for octants, fills shapeList of octants

        :param boundingVolume: bounding box that will be divided into octants by this function
        :param shapeList: List of shapes that intersect with bb
        """
        self.octants = []
        self.shapeList = shapeList
        self.boundingVolume = boundingVolume
        self.parentNode = parentNode
        return

    def initializeOctants(self):
        # create octants
        for n in range(8):
            self.octants.append(OctreeNode(BoundingVolume(single=True),[], self))

        halfXLen = self.boundingVolume.xLenBB / 2
        halfYLen = self.boundingVolume.yLenBB / 2
        halfZLen = self.boundingVolume.zLenBB / 2

        # configure octants v2 coordinates
        # because all octants have the same size we can create one quick variable to make BBv2 to BBv8

        self.octants[0].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([0, 0, 0])
        self.octants[1].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([halfXLen, 0, 0])
        self.octants[2].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([halfXLen, halfYLen, 0])
        self.octants[3].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([0, halfYLen, 0])

        self.octants[4].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([0, 0, halfZLen])
        self.octants[5].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([halfXLen, 0, halfZLen])
        self.octants[6].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([halfXLen, halfYLen, halfZLen])
        self.octants[7].boundingVolume.BBv2 = self.boundingVolume.BBv2 + np.array([0, halfYLen, halfZLen])

        # calculate v8 of octants and finalize bounding box
        BBv2ToBBv8 = np.array([halfXLen, halfYLen, halfZLen])

        for n in range(8):
            self.octants[n].boundingVolume.BBv8 = self.octants[n].boundingVolume.BBv2 + BBv2ToBBv8
            self.octants[n].boundingVolume.finalizeAABB()


        # go through shapeList and check if a shape intersects with our octants
        for shapeNr in range(len(self.shapeList)):
            for octNr in range(8):
                if self.octants[octNr].boundingVolume.intersectsWithBoundingVolume(self.shapeList[shapeNr]):
                    self.octants[octNr].shapeList.append(self.shapeList[shapeNr])
        return
    def isInitialized(self):
        """
        Returns true if octants of this node where previously initialized
        Implies that parent node has divided space enough that further division will contribute nothing
        :return:
        """
        return self.octants != 0 and len(self.octants) > 0

    def uninitialize(self):
        """
        Removes initialized octants
        :return:
        """
        self.octants = 0
        return



class BoundingVolume():
    """
    Represents an axis aligned bounding box in global coordinates
    Aligned to coordinate system in camera class

            v1--------v4
           .|        .|
          . |       . |
         .  |      .  |
        v2--------v3  |
        |   |     |   |
        |   |     |   |
        |   |     |   |
        |   v5----|---v8
        |  .      | .
        |.        |.
        v6--------v7
        Points on the box
    """
    def __init__(self, single=False):
        """
        :param single: Set to true when BoundingVolume is not part of a shape
                        When false BBv2 and BBv8 have to be set manually and
                        finalizeAABB has to be called after that
        """
        self.BBv1 = np.zeros(3)
        self.BBv2 = np.zeros(3)
        self.BBv3 = np.zeros(3)
        self.BBv4 = np.zeros(3)
        self.BBv5 = np.zeros(3)
        self.BBv6 = np.zeros(3)
        self.BBv7 = np.zeros(3)
        self.BBv8 = np.zeros(3)

        if not single:
            self.calcAABB()

        # length of bounding box along axis
        self.xLenBB = 0
        self.yLenBB = 0
        self.zLenBB = 0

        if not single:
            self.finalizeAABB()

        return

    def finalizeAABB(self):
        """
        Calculates the rest of BBv... variables and x,y,zLenBB
        :return:
        """

        # sanity check of BBv1 and BBv8
        if self.BBv2[0] > self.BBv8[0] or self.BBv2[1] > self.BBv8[1] or self.BBv2[2] > self.BBv8[2]:
            raise Exception()

        # calculate lengths of parent bounding boxes
        self.xLenBB = self.BBv8[0] - self.BBv2[0]
        self.yLenBB = self.BBv8[1] - self.BBv2[1]
        self.zLenBB = self.BBv8[2] - self.BBv2[2]

        #calculate rest of BBv...
        self.BBv1 = self.BBv2 + np.array([0, 0, self.zLenBB])
        self.BBv3 = self.BBv2 + np.array([0, self.yLenBB, 0])
        self.BBv4 = self.BBv2 + np.array([0, self.yLenBB, self.zLenBB])
        self.BBv5 = self.BBv2 + np.array([self.xLenBB, 0, self.zLenBB])
        self.BBv6 = self.BBv2 + np.array([self.xLenBB, 0, 0])
        self.BBv7 = self.BBv2 + np.array([self.xLenBB, self.yLenBB, 0])
        return

    def calcAABB(self):
        """
        Has no be overwriten by each child class. Calculates BBv2 and BBv8
        :return:
        """
        raise NotImplementedError()

    def isPointInBoundingVolume(self, boundingVol, pnt):
        """
        Checks if a point is inside a Scene.BoundingVolume
        :param boundingVol:  Scene.BoundingVolume
        :param pnt: numpy array (coordiantes)
        :return:
        """
        return pnt[0] >= boundingVol.BBv2[0] and pnt[0] <= boundingVol.BBv6[0] and \
        pnt[1] >= boundingVol.BBv2[1] and pnt[1] <= boundingVol.BBv3[1] and \
        pnt[2] >= boundingVol.BBv2[2] and pnt[2] <= boundingVol.BBv1[2]

    def _intersectBoundingVolumes(self, bndVol1, bndVol2):
        """
        Checks if all or some points of bndVol2 are in bndVol2
        When bndVol2 contains bndVol1 no intersection will be found, use in reverse to find these
        :param bndVol1: Scene.BoundingVolume
        :param bndVol2: Scene.BoundingVolume
        :return: True if bndVol2 intersects bndVol1
        """
        return self.isPointInBoundingVolume(bndVol1, bndVol2.BBv1) or \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv2) or  \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv3) or \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv4) or \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv5) or \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv6) or \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv7) or \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv8)

    def intersectsWithBoundingVolume(self, bndVol):
        """
        Checks if this bounding volume intersects with given one
        :param bndVol: Scene.BoundingVolume
        :return:
        """
        return self._intersectBoundingVolumes(self, bndVol) or self._intersectBoundingVolumes(bndVol, self)