#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:26:46 2017

@author: lessig
"""

import numpy as np


class Ray :
    maxRayLength = 10000.0


    def __init__(self, oa = np.array([0,0,0]), da = np.array([1,0,0])) :
        self.o = oa
        self.d = da
        self.t = np.float64(Ray.maxRayLength)
        self.firstHitShape = 0 # will be set to the object the ray hit closest
        return


    def hasHitSomething(self) :
        # when a ray hits an object, the intersection method will shorten the ray by altering the t parameter
        return self.t < Ray.maxRayLength

        
        
