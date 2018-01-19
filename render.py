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
from datetime import datetime, timezone
import os



def createScene() :
    
    scene = Scene()

    scene.objects.append(
        Sphere(np.array([0.0, 0.0, 3.0]), 1.0, [1, 1, 1])
    )

    scene.objects.append(
        Sphere(np.array([0.0, -1.0, 3.0]), 2.0, [0, 0, 1])
    )

    scene.objects.append(
        Triangle(np.array([0.0,2.0,4.0]),np.array([3.0,2.0,3.0]),np.array([3.0,5.0,4.0]), [1,0,0])
    )

    scene.lights.append(
        SphereLight(np.array([2.0, 0, 3.0]), 0.1, #position, radius
                    [1, 1, 1], 10) # light color, light intensity
    )
    scene.lights.append(
        SphereLight(np.array([-2.0, 0, 3]), 0.1, #position, radius
                    [1, 1, 1], 5) # light color, light intensity
    )
    scene.lights.append(
        SphereLight(np.array([2.0, 0, -2]), 0.5,  # position, radius
                    [1, 1, 1], 5)  # light color, light intensity
    )

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
        print("\rRender Progress", np.floor(((ix*res_y + iy) / totalPixels) * 100), "%       ", end='', flush=True)

    return cam.image
    
#colors are RGB 0 to 1

width = 512
height = 512

integrator = MISIntegrator()
scene = createScene()

im = render( width, height, scene, integrator)



# save in original resolution
# https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
dpi = 80
figsize = width / float(dpi), height / float(dpi)

# Create a figure of the right size with one axes that takes up the full figure
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])

# Hide spines, ticks, etc.
ax.axis('off')

# Display the image.
ax.imshow(im, interpolation='nearest')

# Add something...

# Ensure we're displaying with square pixels and the right extent.
# This is optional if you haven't called `plot` or anything else that might
# change the limits/aspect.  We don't need this step in this case.
ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

directory = "generatedImages/"
if not os.path.exists(directory):
    os.makedirs(directory)

fig.savefig(directory + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + ".png", transparent=True)
plt.show()






