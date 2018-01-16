#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:42 2017

@author: lessig
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
    
    def intersect(self, ray):
        raise NotImplementedError()
        
        