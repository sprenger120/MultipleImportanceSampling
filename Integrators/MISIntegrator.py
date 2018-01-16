from Integrators.integrator import Integrator
from ray import Ray
import numpy as np

class MISIntegrator(Integrator):

    sampleCount = 100

    def ell(self, scene, ray):
        if (scene.intersectObjects(ray)):
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


            return ray.firstHitShape.color

        return [0,0,0] # no intersection so we stare into the deep void



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
