#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
"""


"""
Axis Aligned Bounding Box Coordinates (Global)
        
         
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
        


"""

import util as util

from abc import ABCMeta

class Shape :
    __metaclass__ = ABCMeta
    
    objectid = 0
    
    def __init__(self, color) :
        Shape.objectid += 1
        if not util.isColor(color) :
            raise Exception()

        self.color = color
        self.BBv1 = np.zeros(3)
        self.BBv2 = np.zeros(3)
        self.BBv3 = np.zeros(3)
        self.BBv4 = np.zeros(3)
        self.BBv5 = np.zeros(3)
        self.BBv6 = np.zeros(3)
        self.BBv7 = np.zeros(3)
        self.BBv8 = np.zeros(3)
        self.calcAABB()

    def intersect(self, ray):
        raise NotImplementedError()

    def calcAABB(self):
        raise NotImplementedError()
        
        