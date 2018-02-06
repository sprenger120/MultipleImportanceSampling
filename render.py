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
from Shapes.Triangle import Triangle
from Shapes.sphere import Sphere
from camera import Camera
import webbrowser
from multiprocessing import Process, Manager


# x5 Render time, anti aliasing
enableSubPixelRendering = False

timeUsedSec = 0


def render( globalRes_x, globalRres_y, x0, y0, x1, y1, threadId, return_dict) :
    global timeUsedSec

    # sanity check window to render
    if x0 >= x1 or y0 >= y1:
        raise Exception()

    print("Rendering Worker #",threadId,": [",x0,",",y0,"] to [", x1, ",", y1,"]")


    integrator = MISIntegrator()
    scene = CornellBox()

    # only thread one is reporting
    if threadId == 0:
        print("\n")
        if enableSubPixelRendering:
            print("Subpixel Rendering enabled (Render time x5)")

    cam = Camera( globalRes_x, globalRres_y)
    totalPixels = (x1-x0) * (y1-y0)

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
    for ix in range(x0,x1) :
        rowTimeSec = time.process_time()
        for iy in range(y0,  y1) :

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

        if threadId == 0:
            calculatedPixels = (ix-x0) * (y1-y0) + (iy-y0)
            rowTimeSec = time.process_time() - rowTimeSec
            timePerPixelSec = rowTimeSec / (y1-y0)
            remainingTimeSec = timePerPixelSec * (totalPixels - calculatedPixels)
            timeUsedSec = time.process_time() - usedTime


            print("\rProgress: ", np.floor((calculatedPixels / totalPixels) * 100),"%",
                  "Time Used: ", util.formatSeconds(timeUsedSec),
                  "ETA: ", util.formatSeconds(remainingTimeSec),
                  "Time per Pixel:%6.1fms" % (timePerPixelSec * 1000),
                  end='', flush=True)

    return_dict[threadId] = cam.image
    return
    #return cam.image

    
#colors are RGB 0 to 1


# distributed-multicore rendering
# each renderer thread has to know how big the resulting image will be
# in addition to that it gets its slice of the picture to render

globalWidth = 512
globalHeight = 512


# to support rendering on multiple computers we define a slice that this
# render instance may divide among its cpu cores

clientSliceX0 = 0
clientSliceY0 = 0
clientSliceX1 = globalWidth
clientSliceY1 = globalHeight


coresToUse = 4


finishedImage = np.zeros((globalWidth, globalHeight, 3), dtype=np.float)


# start processes
# for best results please use even core count

processes = []
manager = Manager()
return_dict = manager.dict()

# each core gets to render an equally big slice of the picture
sliceWidth = np.int(globalWidth / coresToUse * 2)
sliceHeight = np.int(globalHeight / coresToUse * 2)
for x in range(0, globalWidth, sliceWidth):
    for y in range(0, globalHeight, sliceHeight):
        processes.append(Process(target=render,
                                 args=(globalWidth, globalHeight, x, y, x + sliceWidth, y + sliceHeight, len(processes), return_dict)))


for n in range(len(processes)):
    processes[n].start()

for n in range(len(processes)):
    processes[n].join()



generatedImages = return_dict.values()
currentProcess = 0


for x in range(0, globalWidth, sliceWidth):
    for y in range(0, globalHeight, sliceHeight):
        for procX in range(x,x+sliceWidth):
            for procY in range(y, y+sliceHeight):
                finishedImage[procX, procY, :] = generatedImages[currentProcess][procX][procY]
        currentProcess += 1




#
# Generate Time Statistics
#
"""
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
"""



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

scipy.misc.toimage(finishedImage, cmin=0.0, cmax=1).save(filename)

# open with default image application
webbrowser.open(filename)