import numpy as np
import time
import copy
from ray import Ray

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
    OCTREE_COORDINATE_MAX = 100.0

    # Maximum divisions
    MAX_DEPTH = 10

    def __init__(self, shapeArray, coordinateOffset = np.zeros(3)):
        """
        Creates octree from given array of shapes. See create() for more information
        :param shapeArray: [] of Shapes.shape derivates
        :param coordinateOffset: np array with coordinates
                                It is a good idea to minimize the amount of shapes that
                                can't be sorted into octree leaves. When your scene is right in 0,0
                                there is a high chance for many shapes to be unsortable. To fix that you can
                                move the octree slightly to another side so that your scene is within an root octant
        """
        self.rootBoundingVolume = BoundingVolume(single=True)
        self.rootBoundingVolume.BBv2 = np.array([-Octree.OCTREE_COORDINATE_MAX,
                                                -Octree.OCTREE_COORDINATE_MAX,
                                                -Octree.OCTREE_COORDINATE_MAX])
        self.rootBoundingVolume.BBv8 = np.array([Octree.OCTREE_COORDINATE_MAX,
                                                Octree.OCTREE_COORDINATE_MAX,
                                                Octree.OCTREE_COORDINATE_MAX])

        self.rootBoundingVolume.BBv2 += coordinateOffset
        self.rootBoundingVolume.BBv8 += coordinateOffset
        self.rootBoundingVolume.finalizeAABB()

        self.rootNode = 0
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

        val =  self._create(self.rootNode, 1)

        print("Creation Done took: %2.1fms" % ((time.process_time() - t0) * 1000),
              " Root unsortable shape size: ", len(self.rootNode.shapeList))

        return val

    def _create(self, node, recursionLevel):
        if recursionLevel > Octree.MAX_DEPTH:
            return

        node.initializeOctants()


        # initialize all octants of this node
        #for n in range(8):
        #    node.octants[n].initializeOctants()
            # remove octant with no elements
            #if node.octants[n].elementsWithin == 0:
            #    node.octants[n].uninitialize()

        # divide that space further
        for n in range(8):
            if node.octants[n].elementsWithin > 0:
                self._create(node.octants[n], recursionLevel + 1)
        return


    def intersect(self, ray):
        self._intersect(ray, self.rootNode)
        #print(ray.t)
        return ray.t < Ray.maxRayLength

    def _intersect(self, ray, node):
        # intersect unsortable objects
        for n in range(len(node.shapeList)):
            ray.intersecCount += 1
            #if node.shapeList[n].intersect(ray):
            #    ray.firstHitShape = node.shapeList[n]

        # if we are not initialized no need to search through octants
        if not node.isInitialized():
            return

        # check which octant our ray is intersecting
        # if so dive deeper into it
        # each intersection is shortening the ray so we always have the nearest object
        for n in range(8):
            if node.octants[n].boundingVolume.intersectsWithRay(ray):
                self._intersect(ray, node.octants[n])

class OctreeNode():
    """
    Represents one octree node. Divides given bounding volume into 8 smaller pieces (octants)
    Contains:
        - 8 Octants
        - List of shapes that have to be checked when something intersects with the boundingVolume of this node
            - Either objects that can be fully enclosed within this boundingVolume
                (Those objects may be moved further down into octants)
            - Objects that can't be fully enclosed by bounding volume but still intersect with it
                (will stay in this node)
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

        # todo replace this with dynamic lookup how many elements child nodes have
        # count how many elements are sorted within octants and those who just clip this bounding box
        # we store this because when space is further divided elements are taken from our shapeList
        # and we no longer can detmine the object amount without going through every octant recursively
        self.elementsWithin = len(shapeList)
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


        # go through shapeList and check if a shape is fully within an octant and move it to the octants shapeList
        # because its impossible for it to be in another octant
        # all shapes left over are either outside of global octree world size (only happens in root node)
        # or intersect with more than one
        # in that case we keep it in this node

        #backwards because we are deleting
        for shapeNr in range(len(self.shapeList)-1, -1, -1):
            for octNr in range(8):
                if self.octants[octNr].boundingVolume.isBVInSelf(self.shapeList[shapeNr]):
                    self.octants[octNr].shapeList.append(self.shapeList.pop(shapeNr))
                    break

        #recalculate elementsWithin
        for octNr in range(8):
            self.octants[octNr].elementsWithin = len(self.octants[octNr].shapeList)
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
        # a little help with intersectRayWithPlane usage
        if isinstance(pnt, int):
            return False

        return pnt[0] >= boundingVol.BBv2[0] and pnt[0] <= boundingVol.BBv6[0] and \
        pnt[1] >= boundingVol.BBv2[1] and pnt[1] <= boundingVol.BBv3[1] and \
        pnt[2] >= boundingVol.BBv2[2] and pnt[2] <= boundingVol.BBv1[2]

    def isBVWithinBV(self, bndVol1, bndVol2):
        """
        Checks if all points of bndVol2 are in bndVol1

        :param bndVol1: Scene.BoundingVolume
        :param bndVol2: Scene.BoundingVolume
        :return: True when bndVol2 is fully enclosed by bndVol1
        """
        return self.isPointInBoundingVolume(bndVol1, bndVol2.BBv2) and  \
        self.isPointInBoundingVolume(bndVol1, bndVol2.BBv8)

    def isBVInSelf(self, bndVol):
        return self.isBVWithinBV(self, bndVol)

    def intersectRayWithPlane(self, plane_o, plane_normal, ray, t_mul=1.000000):
        # assuming every direction vector is unit long
        a = np.dot(ray.d, plane_normal)
        # parallel to plane
        if a < 0.00001 and a > -0.00001:
            return 0
        t = np.dot((plane_o-ray.o), plane_normal) / a
        return ray.o + ray.d * (t * t_mul)

    def intersectsWithRay(self, ray):
        # if ray starts in bounding volume we know it intersects
        if self.isPointInBoundingVolume(self, ray.o):
            return True

        # intersect ray with every faces plane and see if intersection point
        # use a bit of t_mul to move intersection point behind the intersection plane to
        # avoid float wierdness
        # assuming bounding box is not smaller than 1-t_mul


        # face v1-v2-v3-v4
        if self.isPointInBoundingVolume(self,
                                        self.intersectRayWithPlane(self.BBv1, np.array([-1.0, 0.0, 0.0]), ray)):
            return True
        
        # face v2-v3-v6-v7
        if self.isPointInBoundingVolume(self,
                                        self.intersectRayWithPlane(self.BBv2, np.array([0,0,-1]), ray)):
            return True
        # face v1-v2-v5-v6
        if self.isPointInBoundingVolume(self,
                                        self.intersectRayWithPlane(self.BBv1, np.array([0, -1, 0]), ray)):
            return True

        """
        # face v5-v6-v7-v8
        if self.isPointInBoundingVolume(self,
                                        self.intersectRayWithPlane(self.BBv5, np.array([1,0,0]), ray)):
            return True
        
        # face v1-v5-v8-v4
        if self.isPointInBoundingVolume(self,
                                        self.intersectRayWithPlane(self.BBv1, np.array([0,0,1]), ray)):
            return True

       

        # face v4-v3-v7-v8
        if self.isPointInBoundingVolume(self,
                                        self.intersectRayWithPlane(self.BBv4, np.array([0,1,0]), ray)):
            return True
        """
        return False