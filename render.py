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




def render( globalRes_x, globalRres_y, x0, y0, x1, y1, threadId, return_dict) :
    timeUsedSec = 0

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