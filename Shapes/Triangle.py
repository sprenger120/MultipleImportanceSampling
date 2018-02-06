#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Shapes.shape import Shape
from ray import Ray
import time

# vertices are always given in CW order


class Triangle(Shape):
    #intersectTimeSec = 0
    #intersectCount = 1

    def __init__(self,v1,v2,v3, color):
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.v3 = np.array(v3)

        self.mat = np.zeros((4, 4))

        self.mat[3][0] = 1
        self.mat[3][1] = 1
        self.mat[3][2] = 1
        self.mat[3][3] = 0
        super().__init__(color)

    """def intersectTimed(self, ray):
        Triangle.intersectCount += 1
        t0 = time.process_time()
        val = self.intersectionMoellerTrumbore(ray)
        Triangle.intersectTimeSec += time.process_time() - t0
        return val
    """

    def intersectionBarycentrian(self, ray):
        ray.d = ray.d / np.linalg.norm(ray.d)
        """
        A(0,0)=a[0];    A(0,1)=b[0];     A(0,2)=c[0];A       (0,3)=-d[0];
        A(1,0)=a[1];A(1,1)=b[1];A(1,2)=c[1];A(1,3)=-d[1];
        A(2,0)=a[2];A(2,1)=b[2];A(2,2)=c[2];A(2,3)=-d[2];
        A(3,0)=1   ;A(3,1)=1   ;A(3,2)=1   ;A(3,3)=0;
        """

        self.mat[0][0] = self.v1[0]
        self.mat[0][1] = self.v2[0]
        self.mat[0][2] = self.v3[0]
        self.mat[0][3] = -ray.d[0]

        self.mat[1][0] = self.v1[1]
        self.mat[1][1] = self.v2[1]
        self.mat[1][2] = self.v3[1]
        self.mat[1][3] = -ray.d[1]

        self.mat[2][0] = self.v1[2]
        self.mat[2][1] = self.v2[2]
        self.mat[2][2] = self.v3[2]
        self.mat[2][3] = -ray.d[2]

        matr = np.matrix(self.mat)
        # singular matrix -> no inverse
        if np.linalg.det(matr) < 0.00001:
            return False

        # mat = np.linalg.inv(mat)
        b = np.array([ray.o[0], ray.o[1], ray.o[2], 1])

        x = np.dot(matr.I, b).getA()[0]

        if x[0] < 0 or x[0] > 1 or x[1] < 0 or x[1] > 1 or x[2] < 0 or x[2] > 1:
            return False
        # print(x)
        if x[3] < 0 or x[3] > Ray.maxRayLength:
            return False

        ray.t = x[3]
        """
        uvw[0] = x[0];
        uvw[1] = x[1];
        uvw[2] = x[2];
        lambda = x[3];
        """

        return True

    #Moeller-Trumbore
    def intersect(self, ray):
        epsilon = 0.0000001
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1

        h = np.cross(ray.d, edge2)
        a = np.dot(edge1, h)
        if a > -epsilon and a < epsilon:
            return False
        f = 1/a
        s = ray.o - self.v1
        u = f * (np.dot(s,h))
        if u < 0.0 or u > 1.0:
            return False
        q = np.cross(s, edge1)
        v = f * np.dot(ray.d, q)
        if v < 0 or u+v > 1:
            return False

        t = f * np.dot(edge2, q)
        if t > epsilon:
            ray.t = t
            return True
        else:
            return False

    def calcAABB(self):
        minZ = self.min(self.v1[2], self.v2[2], self.v3[2])
        maxZ = self.max(self.v1[2], self.v2[2], self.v3[2])

        minY = self.min(self.v1[1], self.v2[1], self.v3[1])
        maxY = self.max(self.v1[1], self.v2[1], self.v3[1])

        minX = self.min(self.v1[0], self.v2[0], self.v3[0])
        maxX = self.max(self.v1[0], self.v2[0], self.v3[0])

        #self.BBv1 = np.array([minX,minY,maxZ])
        self.BBv2 = np.array([minX,minY,minZ])
        #self.BBv3 = np.array([minX,maxY,minZ])
        #self.BBv4 = np.array([minX,maxY,maxZ])

        #self.BBv5 = np.array([maxX,minY,maxZ])
        #self.BBv6 = np.array([maxX,minY,minZ])
        #self.BBv7 = np.array([maxX,maxY,minZ])
        self.BBv8 = np.array([maxX,maxY,maxZ])

        return

    def min(self, a, b, c):
        if a <= b :
            if a <= c:
                return a
            else:
                return c
        else:
            if b <= c:
                return b
            else:
                return c

    def max(self, a, b, c):
        if a >= b:
            if a >= c:
                return a
            else:
                return c
        else:
            if b >= c:
                return b
            else:
                return c