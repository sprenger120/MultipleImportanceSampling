#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from shape import Shape

# vertices are always given in CW order


class Triangle(Shape):
    def __init__(self,
                 x1, y1, z1,
                 x2,y2, z2,
                 x3,y3, z3, color):

        super().__init__(color)
        # todo better notation for byzentian coordinates / intersection
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3


    def intersect(self, ray):
        #todo implement
        return True


