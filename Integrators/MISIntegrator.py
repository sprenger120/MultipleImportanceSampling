from Integrators.integrator import Integrator
from ray import Ray
import numpy as np
import util as util

class MISIntegrator(Integrator):

    sampleCount = 100

    def ell(self, scene, ray):
        if scene.intersectLights(ray) or scene.intersectObjects(ray) :
            # intersection point where object was hit
            intersPoint = ray.o + ray.d*ray.t

            # only for spheres
            normal = np.linalg.norm(intersPoint)


            #todo mix color
            #todo incorporate light intensity to color mixing
            #todo precalculate some more values (normal on intersection point)
            #todo weight sampling methods against each other

            """
            mixing light:
            
            every object has a material that defines  
            reflectance, shininess and color
            
            when light ray falls onto object the angle between intersection
            normal an light will be taken into account by lamberts cosine law
            for specular spots phong can be used 
            
            
            light colors of multiple lights are added and combined by their intensity 
            --> same weighing strategy as used in MIS
            
            combined light intensity is to be calculated seen from the intersection point of the sphere
            light intensity drops off with 1/sqrt(distance to object) and then simply added
            
        
                    
            the mixed lights color is mixed with the objects color and then multiplied by the combined intensity
            """
            val = self.RandomStupidSampling(intersPoint, ray, scene)
            return val
            # return ray.firstHitShape.color

        return [0,0,0] # no intersection so we stare into the deep void



    def BRDFSampling(self, intersectionPoint, ray, scene) :
        #todo
        return 0

    def LightSoureAreaSampling(self, intersectionPoint, ray, scene):
        #todo
        return 0

    def RandomStupidSampling(self, intersPoint, ray, scene):
        # all the light hitting our intersection point
        # this value is later normalized with the sample count
        # before that its just the sum of incoming light
        aquiredLightSum = 0

        # Array of light intensity value that goes into the integrator and the color
        # of the light [ (lightIntensity,[R,G,B]), ... ]
        aquiredLightsIntensity = np.zeros(MISIntegrator.sampleCount)
        aquiredLightsColor = np.zeros((MISIntegrator.sampleCount, 3))

        # filled out elements in the array
        aquiredLightsCount = 0

        # integrate over sphere using monte carlo
        for sampleNr in range(MISIntegrator.sampleCount):
            lightSenseRay = Ray(intersPoint)

            # generate random direction
            # *2 -1 to generate negatives too
            randomDirection = np.random.random(3) * 2 -1
            lightSenseRay.d = randomDirection

            # send ray on its way
            if scene.intersectLights(lightSenseRay) :
                # weigh light intensity by various factors
                aquiredLight = lightSenseRay.firstHitShape.lightIntensity
                # lambert light model (cos weighting)

                # todo  this line is incorrect;   we need angle between intersection normal and
                # ray being shot out
                aquiredLight *= np.cos(randomDirection[1])

                aquiredLightSum += aquiredLight

                aquiredLightsIntensity[aquiredLightsCount] = aquiredLight
                aquiredLightsColor[aquiredLightsCount] = lightSenseRay.firstHitShape.lightColor
                aquiredLightsCount += 1

        combinedLightColor = np.zeros(3)

        # avoid / 0 when no light was aquired
        if aquiredLightSum > 0 :
            # calculate pixel color
            # first calculate the color of the light hitting the shape
            # light that is more intense has more weight in the resulting color

            for n in range(aquiredLightsCount) :
                combinedLightColor += aquiredLightsColor[n] * (aquiredLightsIntensity[n] / aquiredLightSum)

            # should not be necessary
            combinedLightColor = util.clipColor(combinedLightColor)

            # normalize light
            aquiredLightSum /= MISIntegrator.sampleCount

        # combine light color and object color + make it as bright as light that falls in
        return ray.firstHitShape.color * combinedLightColor * aquiredLightSum

    """
    Mixes objects color with light color
    """
    def mixColor(self, objectColor, lightColor):
        #todo implement
        """
        //Mixes the color of an object with the color of the light shining at it
        //expects all colors to be between 0 and 1
        //Assuming that all light only consists of red, green, blue
        //Then a green object is one that absorbs all red and green light and reflects red light
        //Therefore when a green object is illuminated with red light it will be black because it
        //absorbs all red light;  same goes for blue light
        //So to realistically mix object and light color we define that  1-object color is the
        //absorption factor of the incoming light
        //So to have the above mentioned effect we calculate
        // lightColor -  (1-objectColor) to incorporate the absorption factor
        Vec3d PhongMaterial::mixColor(const Vec3d &baseColor, const Vec3d &lightColor) {
            return Vec3d(baseColor.x() - 1 + lightColor.x(),
                         baseColor.y() - 1 + lightColor.y(),
                         baseColor.z() - 1 + lightColor.z());
        }
        """
        return
