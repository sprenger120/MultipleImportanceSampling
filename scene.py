#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:54:07 2017

@author: lessig
"""

# implement cornell box

class Scene :
    
    def __init__(self) :
        self.objects = []
        self.lights = []


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