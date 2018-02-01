#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Requires:
    - mathplotlib
    - scipy
"""

import os
import time
from datetime import datetime, timezone

import numpy as np
import scipy.misc

import util as util
from Integrators.MISIntegrator import MISIntegrator
from Scene.Scenes.CornellBox import CornellBox
from Shapes.Lights.Lights import SphereLight, TriangleLight
from Shapes.Triangle import Triangle
from Shapes.sphere import Sphere
from camera import Camera
import webbrowser



enableSubPixelRendering = False # x5 Render time


# time spent rendering
timeUsedSec = 0


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
              "Time Used: ", util.formatSeconds(timeUsedSec),
              "ETA: ", util.formatSeconds(remainingTimeSec),
              "Time per Pixel:%6.1fms" % (timePerPixelSec * 1000),
              end='', flush=True)

    return cam.image

    
#colors are RGB 0 to 1

width = 256
height = 256

integrator = MISIntegrator()
scene = CornellBox()

#cProfile.run('im = render( width, height, scene, integrator)')
image = render(width, height, scene, integrator)




#
# Generate Time Statistics
#

overall = timeUsedSec
times = [
    ["Triangle Intersect", Triangle.intersectTimeSec],
    ["Sphere Intersect", Sphere.intersectTimeSec],
    ["Trace Prep", MISIntegrator.TracePrepTimeSec],
    ["Ray Generation", MISIntegrator.RayGenTimeSec],
    ["Color Generation", MISIntegrator.ColorGenTimeSec],

    ["Other", overall] # don't remove or edit
    ]


print("\n----------------------- Render Statistic -----------------------")
print("Overall Time: %2.1fs" % overall)

for n in range(len(times)):
    print(times[n][0], ": %2.1fs" % times[n][1], " %2.1f" % ((times[n][1] / overall) * 100), "%" )
    if n < len(times) - 1:
        times[len(times)-1][1] -= times[n][1]

print("Triangle Intersect Count: ", Triangle.intersectCount, " intersect speed: %6.2f µs" % ((Triangle.intersectTimeSec / Triangle.intersectCount) * 1000000))
print("Sphere Intersect Count: ", Sphere.intersectCount, " intersect speed: %6.2f µs" % ((Sphere.intersectTimeSec / Sphere.intersectCount) * 1000000))


print("\n----------------------- Render Statistic -----------------------")


#
# Save picture and show
#

directory = "generatedImages"
if not os.path.exists(directory):
    os.makedirs(directory)

filename = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") \
           + "_" + str(MISIntegrator.sampleCount) + "Samples_RenderTime_" + util.formatSeconds(timeUsedSec)
if enableSubPixelRendering:
    filename += "_SubpixelRendering"

filename += ".png"
filename = os.path.join(directory,filename)

scipy.misc.toimage(image, cmin=0.0, cmax=1).save(filename)

# open with default image application
webbrowser.open(filename)