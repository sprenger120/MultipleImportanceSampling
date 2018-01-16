from Integrators.integrator import Integrator
from ray import Ray
import numpy as np

class MISIntegrator(Integrator):

    sampleCount = 100

    def ell(self, scene, ray):
        if (scene.intersectObjects(ray)):
            # intersection point where object was hit
            intersPoint = ray.o + ray.d*ray.t


            #todo mix color
            #todo incorporate light intensity to color mixing
            #todo precalculate some more values (normal on intersection point)
            #todo weight sampling methods against each other


            return ray.firstHitShape.color


        return [0,0,0] # no intersection so we stare into the deep void

#todo return resulting color (with light intensity incorporated already ?)

    def BRDFSampling(self, intersectionPoint, ray, scene) :
        #todo
        return 0

    def LightSoureAreaSampling(self, intersectionPoint, ray, scene):
        #todo
        return 0

    def RandomStupidSampling(self, intersPoint, ray, scene):
        #todo
        # integrate over sphere using monte carlo
        for sampleNr in range(MISIntegrator.sampleCount):
            lightSenseRay = Ray(intersPoint)

            # generate random direction
            randomDirection = np.rand(3)
        return 0
