#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:48 2017

@author: lessig
"""

import numpy as np
import matplotlib.pyplot as plt

import util as util


from ray import Ray
from camera import Camera
from sphere import Sphere
from scene import Scene
from basic_integrator import BasicIntegrator
from MISIntegrator import MISIntegrator


def createScene() :
    
    scene = Scene()
    
    sphere = Sphere( np.array([0.0, 0.0, 3.0]), 1.0, [0,1,0])
    scene.objects.append( sphere)
    
    return scene


def createCornellBox():
    # todo
    return



def render( res_x, res_y, scene, integrator) :
    
    cam = Camera( res_x, res_y)
    
    for ix in range( res_x) :
        for iy in range( res_y) :

            r = cam.generateRay( ix, iy)

            ellval = integrator.ell( scene, r)
            # always clip color so that imshow interpretes as rgb 0.0 - 1.0
            # clip color will also check color valididty and raise exception if necessary
            cam.image[ix,iy, :] = util.clipColor(ellval)

    return cam.image
    
#colors are RGB 0 to 1

integrator = MISIntegrator()
scene = createScene()

im = render( 512, 512, scene, integrator)

plt.imshow( im)
#plt.colorbar()
plt.show()


