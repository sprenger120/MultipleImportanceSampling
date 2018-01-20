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
from Shapes.Lights.Lights import SphereLight, TriangleLight
from camera import Camera
from scene import Scene
from datetime import datetime, timezone
import os
import time


"""

CCW coordinate direction

"""

enableSubPixelRendering = False # x5 Render time


# time spent rendering
timeUsedSec = 0

def createScene() :
    scene = Scene()

    # can be directly taken from fast triangle viewer
    # only contains triangles
    polyArray = [
        [[0.0, 2.0, -4.0], [3.0, 2.0, -3.0], [3.0, 3.0, -4.0], [1, 1, 1]],

        [[-3.0, 0, -4], [-3, 0, -4], [-3, 3, -6], [1, 1, 1]],
    ]


    # transfer poly array to scene
    for n in range(len(polyArray)) :
        scene.objects.append(
            Triangle(np.array(polyArray[n][0]), np.array(polyArray[n][1]), np.array(polyArray[n][2]), polyArray[n][3])
        )


    scene.objects.append(
        Sphere(np.array([0.0, -2.0, -3.0]), 0.5, [1, 1, 1])
    )

    scene.objects.append(
        Sphere(np.array([1.0, -1.0, 0.0]), 0.5, [0, 0, 1])
    )



    # does not seem to work yet
    """
    scene.lights.append(
        TriangleLight(np.array([-3.0, 0, -4]),np.array([-2.5, 0, -4]),np.array([-2.5, 3, -6]),
                      [1,1,1], 100)
    )
    """

    scene.lights.append(
        SphereLight(np.array([-2.0, 3, 0.0]), 1, #position, radius
                    [1, 1, 1], 8) # light color, light intensity
    )

    scene.lights.append(
        SphereLight(np.array([-2.0, -3, 0]), 1, #position, radius
                    [1, 0, 0], 8) # light color, light intensity
    )

    return scene


def createCoordinateScene() :
    scene = Scene()

    scene.objects.append(
        Sphere(np.array([0, 0.0, 0]), 0.2, [1, 1, 1])
    )

    scene.objects.append(
        Sphere(np.array([1, 0.0, 0]), 0.2, [1, 0, 0])
    )

    scene.objects.append(
        Sphere(np.array([0, 1.0, 0]), 0.2, [0, 1, 0])
    )

    scene.objects.append(
        Sphere(np.array([-1, 0.0, 5]), 0.2, [0, 0, 1])
    )


    scene.lights.append(
        SphereLight(np.array([0, 0, 3.0]), 0.5,  # position, radius
                    [1, 1, 1], 2)  # light color, light intensity
    )


    return scene

def createCornellBox():
    # todo
    return

def formatSeconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)



def render( res_x, res_y, scene, integrator) :
    global timeUsedSec

    print("\n")
    if enableSubPixelRendering:
        print("Subpixel Rendering enabled (Render time x5)")

    cam = Camera( res_x, res_y)
    totalPixels = res_x * res_y

    """
        * Main pixel (without offset)
        # Sub Pixels

        (0,0)    (1, 0)
        *----------#
        |          |
        |  Pixel   |
        |    on    |
        |  Screen  |
        |          |
        #----------#     
        (0, 1)   (1,1)
    """
    usedTime = time.process_time()
    for ix in range( res_x) :
        rowTimeSec = time.process_time()
        for iy in range( res_y) :

            subPixelCount = 1
            ellValSum = np.zeros(3)

            #top left
            r = cam.generateRay(ix, iy)
            ellValSum += integrator.ell(scene, r)

            if enableSubPixelRendering:
                # middle
                r = cam.generateRay(ix + 0.5, iy + 0.5)
                ellValSum += integrator.ell(scene, r)
                subPixelCount += 1

                #top right
                r = cam.generateRay(ix+1, iy)
                ellValSum += integrator.ell(scene, r)
                subPixelCount += 1

                #bottom left
                r = cam.generateRay(ix, iy + 1)
                ellValSum += integrator.ell(scene, r)
                subPixelCount += 1

                # bottom right
                r = cam.generateRay(ix + 1, iy + 1)
                ellValSum += integrator.ell(scene, r)
                subPixelCount += 1

                ellValSum /= subPixelCount

            # always clip color so that imshow interpretes as rgb 0.0 - 1.0
            # clip color will also check color valididty and raise exception if necessary
            cam.image[ix, iy, :] = util.clipColor(ellValSum)

        # some progress and time estimation
        calculatedPixels = ix * res_y + iy
        rowTimeSec = time.process_time() - rowTimeSec
        timePerPixelSec = rowTimeSec / res_y
        remainingTimeSec = timePerPixelSec * (totalPixels - calculatedPixels)
        timeUsedSec = time.process_time() - usedTime


        print("\rProgress: ", np.floor((calculatedPixels / totalPixels) * 100),"%",
              "Time Used: ", formatSeconds(timeUsedSec),
              "ETA: ", formatSeconds(remainingTimeSec),
              "Time per Pixel:%6.1fms" % (timePerPixelSec * 1000),
              end='', flush=True)

    return cam.image

    
#colors are RGB 0 to 1

width = 512
height = 512

integrator = MISIntegrator()
scene = createScene()
#scene = createCoordinateScene()

#im = render( width, height, scene, integrator)

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

filename = directory + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") \
           + "_" + str(MISIntegrator.sampleCount) + "Samples_RenderTime_" + formatSeconds(timeUsedSec)

if enableSubPixelRendering:
    filename += "_SubpixelRendering"

fig.savefig(filename + ".png", transparent=True)
plt.show()






