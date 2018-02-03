#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Shapes.Triangle import Triangle
import numpy as np
from Scene.Octree import Octree

class Scene :
    """

    CCW coordinate direction

    """

    def __init__(self) :
        self.objects = []
        self.lights = []

        self.importGeometry()

        self.octreeObjects = Octree(self.objects)
        self.octreeLights = Octree(self.lights)
        return

    #todo implement octree
    def intersectLights(self, ray):
        return self.intersect(ray, self.lights)

    def intersectObjects(self, ray):
        return self.intersect(ray, self.objects)

    def intersect( self, ray, list) :
        res = False
        for obj in list :
            if obj.intersect( ray) :
                res |= True
                ray.firstHitShape = obj

        return res

    def importPolyArray(self, polyArray):
        """
        Imports polyArray format of FastTriangleViewer

        :param polyArray:
        :return:
        """
        for n in range(len(polyArray)):
            self.objects.append(
                Triangle(np.array(polyArray[n][0]), np.array(polyArray[n][1]), np.array(polyArray[n][2]),
                         polyArray[n][3])
            )
        return

    def importGeometry(self):
        """
        Should be overwritten with scene creation function
        :return:
        """
        raise NotImplementedError()