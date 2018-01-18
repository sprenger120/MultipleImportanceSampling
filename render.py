#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:48 2017

@author: lessig
"""

import matplotlib.pyplot as plt
import numpy as np

import util as util
from Integrators.MISIntegrator import MISIntegrator
from Shapes.sphere import Sphere
from Shapes.Triangle import Triangle
from Shapes.Lights.Lights import SphereLight
from camera import Camera
from scene import Scene
from sys import stdout


def createScene() :
    
    scene = Scene()

    scene.objects.append(
        Sphere(np.array([0.0, 0.0, 3.0]), 1.0, [1, 1, 1])
    )

    scene.objects.append(
        Sphere(np.array([0.0, -1.0, 3.0]), 2.0, [0, 0, 1])
    )

    scene.objects.append(
        Triangle(np.array([0.0,0.0,4.0]),np.array([3.0,0.0,3.0]),np.array([3.0,3.0,4.0]), [1,0,0])
    )

    scene.lights.append(
        SphereLight(np.array([2.0, 0, 3.0]), 0.1, #position, radius
                    [1, 1, 1], 10) # light color, light intensity
    )
   # scene.lights.append(
   #     SphereLight(np.array([-2.0, 0, 3]), 0.1, #position, radius
   #                 [1, 0, 0], 5) # light color, light intensity
  #  )
   # scene.lights.append(
   #     SphereLight(np.array([3.0, 0, 3]), 0.1,  # position, radius
   #                 [0, 1, 0], 10)  # light color, light intensity
   # )

    return scene


def createCornellBox():
    # todo
    return



def render( res_x, res_y, scene, integrator) :
    print("\n")
    cam = Camera( res_x, res_y)
    totalPixels = res_x * res_y
    
    for ix in range( res_x) :
        for iy in range( res_y) :

            r = cam.generateRay( ix, iy)

            ellval = integrator.ell( scene, r)
            # always clip color so that imshow interpretes as rgb 0.0 - 1.0
            # clip color will also check color valididty and raise exception if necessary
            cam.image[ix,iy, :] = util.clipColor(ellval)
        print("\rRender Progress", np.floor(((ix*res_x + iy) / totalPixels) * 100), "%       ", end='', flush=True)

    return cam.image
    
#colors are RGB 0 to 1

integrator = MISIntegrator()
scene = createScene()

im = render( 256, 256, scene, integrator)

plt.imshow( im)
#plt.colorbar()
plt.show()


