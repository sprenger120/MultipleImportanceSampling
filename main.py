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
from render import render

def main():
    # colors are RGB 0 to 1


    # distributed-multicore rendering
    # each renderer thread has to know how big the resulting image will be
    # in addition to that it gets its slice of the picture to render

    globalWidth = 256
    globalHeight = 256

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
    sliceWidth = np.int(globalWidth / np.sqrt(coresToUse))
    sliceHeight = np.int(globalHeight / np.sqrt(coresToUse))
    for x in range(0, globalWidth, sliceWidth):
        for y in range(0, globalHeight, sliceHeight):
            processes.append(Process(target=render,
                                     args=(
                                     globalWidth, globalHeight, x, y, x + sliceWidth, y + sliceHeight, len(processes),
                                     return_dict)))

    for n in range(len(processes)):
        processes[n].start()

    for n in range(len(processes)):
        processes[n].join()

    generatedImages = return_dict.values()
    currentProcess = 0

    for x in range(0, globalWidth, sliceWidth):
        for y in range(0, globalHeight, sliceHeight):
            for procX in range(x, x + sliceWidth):
                for procY in range(y, y + sliceHeight):
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

    filename = datetime.now(timezone.utc).strftime("%Y_%m_%d__%H_%M_%S") \
               + "_" + str(MISIntegrator.sampleCount) + "Samples"

    filename += ".png"
    filename = os.path.join(os.getcwd(), directory, filename)
    print("Saving to:", filename)

    scipy.misc.toimage(finishedImage, cmin=0.0, cmax=1).save(filename)

    # open with default image application
    webbrowser.open(filename)
    return

# fix for windows doing stupid things with multiprocessing
# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ ==  '__main__':
    main()