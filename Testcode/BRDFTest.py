import matplotlib.pyplot as plot
from Materials.BRDF import BRDF
import os
import numpy as np


def integrate0To1(array):
    sliceSize = 1 / len(array)
    value = 0
    for n in range(len(array)):
        value += array[n]*sliceSize
    return value

if __name__ == "__main__":
    brdf = BRDF("/home/sprenger/workspace/MultipleImportanceSampling/Materials/brdf_data/alum-bronze.binary")




    steps = 45
    rawValues = np.zeros((steps,2))
    n = 0

    for i in np.linspace(-np.pi, np.pi, steps):
        rawValues[n] = [i, np.maximum(brdf.lookupValue(1, 1,0,0)[0],0)]
        n += 1


    integr = integrate0To1(rawValues[:, 1])
    print("size before normaliz: ", integr)

    rawValues[:, 1] /= integr
    rawValues[:, 0] = np.linspace(0, 1, steps)


    integr2 = integrate0To1(rawValues[:, 1])
    print("size after normaliz: ", integr2)


    plot.plot(rawValues[:, 0], rawValues[:, 1])
    cumsum = np.cumsum(rawValues[:, 1] * (1/steps))
    #plot.plot(rawValues[:, 0], cumsum)

    """
    samples = np.random.uniform(0,1,999999)

    samplesPerStep = np.zeros(steps)


    for sampleNr in range(len(samples)):
        for i in range(steps):
            if cumsum[i] > samples[sampleNr]:
                samplesPerStep[i] += 1
                break


    plot.scatter(rawValues[:, 0], samplesPerStep)
    """

    plot.show()