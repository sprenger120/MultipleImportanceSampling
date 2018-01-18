from Integrators.integrator import Integrator
from ray import Ray
import numpy as np
import util as util

"""
Angles on the sphere


   seen from above:
      .........
     .       /  .
   .        /     .
   .       /  phi  .
   .      x _______. 0°
   .              .
   . .        . .
       .......
 0 - 180°;  -1 - 1


 seen from the side:
 
  0 - 90°;   0 - 1
                  theta
         ......0°.......
       .       |   /   .
     .         |  /      .
    .          | /        .
   .___________x__________.
"""



class MISIntegrator(Integrator):

    sampleCount = 500

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

            # generate direction of light sense ray shot away from the hemisphere
            # generate theta and phi and map onto the sphere
            # direction is vector from intersection point to point on hemisphere

            theta = np.random.random(1)
            phi = np.random.random(1) * 2 - 1  #*2 -1 for negative numbers

            """
            N... Normal on intersection point,  normalized
            
            
            
            ^ ------->
            |        |
            |        |
            |        |
            |        ^
            N
            
            """






            # *2 -1 to generate negatives too
            #todo fix, negative values should only come from theta angle
            #todo generate theta and phi instead of three coordinates
            randomDirection = np.random.random(3) * 2 - 1
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
    omega: Array with angles (theta, phi) (in radiant)
    normal: Normal of origin of sphere, resulting vector will be adjusted for given normal
            leave default for upright normal 
            coordinate system orientation cam be seen in camera.py
            has to be normalized
    Returns vector from coordinate origin to point on unit sphere where both angles would put it
    """
    def s2tor3(omega, normal = np.array([1, 0, 0])):

        rotX = 0
        rotY = 0
        rotZ = 0
        rotationMatrix = np.array([[],
                                   [],
                                   []])



        r3 = np.zeros(3)
        r3[0] = np.sin(omega[0]) * np.cos(omega[1])
        r3[1] = np.sin(omega[0]) * np.sin(omega[1])
        r3[2] = np.cos(omega[0])
        return r3

    def rotation_matrix_numpy(axis, theta):
        # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
        mat = np.eye(3, 3)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.)
        b, c, d = -axis * np.sin(theta / 2.)

        return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

