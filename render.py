#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Requires:
    - mathplotlib
    - scipy
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
import cProfile
from sys import platform
import scipy.misc


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
        [[5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [5.0, -5.0, 12.0], [1, 1, 1]],  # floor

        [[5.0, 5.0, 0.0], [5.0, 5.0, 12.0], [5.0, -5.0, 12.0], [1, 1, 1]],  # floor

        [[5.0, -5.0, 0.0], [5.0, -5.0, 12.0], [-5.0, -5.0, 0.0], [1, 0, 0]],  # left wall

        [[5.0, -5.0, 12.0], [-5.0, -5.0, 12.0], [-5.0, -5.0, 0.0], [1, 0, 0]],  # left wall

        [[5.0, 5.0, 0.0], [5.0, 5.0, 12.0], [-5.0, 5.0, 0.0], [0, 1, 0]],  # right wall

        [[5.0, 5.0, 12.0], [-5.0, 5.0, 12.0], [-5.0, 5.0, 0.0], [0, 1, 0]],  # right wall

        [[5.0, -5.0, 12.0], [5.0, 5.0, 12.0], [-5.0, 5.0, 12.0], [1, 1, 1]],  # back wall

        [[5.0, -5.0, 12.0], [-5.0, 5.0, 12.0], [-5.0, -5.0, 12.0], [1, 1, 1]],  # back wall

        [[-5.0, -5.0, 0.0], [-5.0, 5.0, 0.0], [-5.0, -5.0, 12.0], [1, 1, 1]],  # ceiling

        [[-5.0, 5.0, 0.0], [-5.0, 5.0, 12.0], [-5.0, -5.0, 12.0], [1, 1, 1]],  # ceiling

    ]
    # transfer poly array to scene
    for n in range(len(polyArray)) :
        scene.objects.append(
            Triangle(np.array(polyArray[n][0]), np.array(polyArray[n][1]), np.array(polyArray[n][2]), polyArray[n][3])
        )

    #scene.lights.append(
    #    SphereLight(np.array([0.0, 0.0, 0.0]), 1,  # position, radius
    #                [1, 1, 1], 1)  # light color, light intensity
    #)

    scene.lights.append(
        TriangleLight(np.array([-5.0, -1.0, 3.0]), np.array([-5.0, -1.0, 4.0]), np.array([-5.0, 1.0, 3.0]),
                      [1, 1, 1], 2)
    )

    scene.lights.append(
        TriangleLight(np.array([-5.0, -1.0, 4.0]), np.array([-5.0, 1.0, 4.0]), np.array([-5.0, 1.0, 3.0]),
                      [1, 1, 1], 2)
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
                    [1, 1, 1], 0.5)  # light color, light intensity
    )



    return scene

def createCornellBox():
    scene=Scene()
    polyArray = [
        [[5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [5.0, -5.0, 10.0], [1, 1, 1]],      # floor

        [[5.0, 5.0, 0.0], [5.0, 5.0, 10.0], [5.0, -5.0, 10.0], [1, 1, 1]],      # floor

        [[5.0, -5.0, 0.0], [5.0, -5.0, 10.0], [-5.0, -5.0, 0.0], [1, 0, 0]],    # left wall

        [[5.0, -5.0, 10.0], [-5.0, -5.0, 10.0], [-5.0, -5.0, 0.0], [1, 0, 0]],  # left wall

        [[5.0, 5.0, 0.0], [5.0, 5.0, 10.0], [-5.0, 5.0, 0.0], [0, 1, 0]],       # right wall

        [[5.0, 5.0, 10.0], [-5.0, 5.0, 10.0], [-5.0, 5.0, 0.0], [0, 1, 0]],     # right wall

        [[5.0, -5.0, 10.0], [5.0, 5.0, 10.0], [-5.0, 5.0, 10.0], [1, 1, 1]],    # back wall

        [[5.0, -5.0, 10.0], [-5.0, 5.0, 10.0], [-5.0, -5.0, 10.0], [1, 1, 1]],  # back wall

        [[-5.0, -5.0, 0.0], [-5.0, 5.0, 0.0], [-5.0, -5.0, 10.0], [1, 1, 1]],   # ceiling

        [[-5.0, 5.0, 0.0], [-5.0, 5.0, 10.0], [-5.0, -5.0, 10.0], [1, 1, 1]],   # ceiling

        [[5.0, 1.0, 2.0], [5.0, 3.0, 2.0], [3.0, 3.0, 2.0], [1, 1, 1]],         # first block

        [[5.0, 1.0, 2.0], [3.0, 3.0, 2.0], [3.0, 1.0, 2.0], [1, 1, 1]],         # first block

        [[5.0, 3.0, 2.0], [5.0, 3.0, 4.0], [3.0, 3.0, 4.0], [1, 1, 1]],         # first block

        [[5.0, 3.0, 2.0], [3.0, 3.0, 4.0], [3.0, 3.0, 2.0], [1, 1, 1]],         # first block

        [[3.0, 1.0, 2.0], [3.0, 3.0, 2.0], [3.0, 3.0, 4.0], [1, 1, 1]],         # first block

        [[3.0, 1.0, 2.0], [3.0, 3.0, 4.0], [3.0, 1.0, 4.0], [1, 1, 1]],         # first block
    ]

    # transfer poly array to scene
    for n in range(len(polyArray)):
        scene.objects.append(
            Triangle(np.array(polyArray[n][0]), np.array(polyArray[n][1]), np.array(polyArray[n][2]), polyArray[n][3])
        )
    """
    scene.lights.append(
        SphereLight(np.array([0.0, 0.0, 0.0]), 1,  # position, radius
                    [1, 1, 1], 16)  # light color, light intensity
    )
    """

    scene.lights.append(
        TriangleLight(np.array([-5.0, -1.0, 3.0]), np.array([-5.0, -1.0, 4.0]), np.array([-5.0, 1.0, 3.0]),
                      [1, 1, 1], 2)
    )

    scene.lights.append(
        TriangleLight(np.array([-5.0, -1.0, 4.0]), np.array([-5.0, 1.0, 4.0]), np.array([-5.0, 1.0, 3.0]),
                      [1, 1, 1], 2)
    )

    return scene

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

width = 256
height = 256

integrator = MISIntegrator()
#scene = createScene()
#scene = createCoordinateScene()
scene = createCornellBox()

#im = render( width, height, scene, integrator)


#cProfile.run('im = render( width, height, scene, integrator)')
im = render( width, height, scene, integrator)




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


"""
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

plt.show()
"""


directory = "generatedImages"
if not os.path.exists(directory):
    os.makedirs(directory)

filename = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") \
           + "_" + str(MISIntegrator.sampleCount) + "Samples_RenderTime_" + formatSeconds(timeUsedSec)
if enableSubPixelRendering:
    filename += "_SubpixelRendering"

filename += ".png"
filename = os.path.join(directory,filename)

scipy.misc.toimage(im, cmin=0.0, cmax=1).save(filename)









