from bisect import bisect_left
from Integrators.integrator import Integrator
from ray import Ray
import numpy as np
import util as util
from Shapes.Triangle import  Triangle
import copy
import time
from Shapes.Lights.LightBase import LightBase
from Shapes.Lights.Lights import TriangleLight, SphereLight
import sobol_seq
from Materials.BRDF import BRDF
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    sampleCount = 32
    defaultHemisphereNormal = np.array([0.0, 0.0, 1.0])

    LightSourceAreaSamplingPdf=0


    def __init__(self):
        self.brdf = BRDF(os.path.join(os.getcwd(),"Materials", "brdf_data", "alum-bronze.binary"))
        return


    def ell(self, scene, ray):
        hitSomething = scene.intersectObjects(ray)
        hitSomething |= scene.intersectLights(ray)

        # we have hit an object
        if  hitSomething :
            ray.t -= 0.001

            if isinstance(ray.firstHitShape, LightBase):
                # we have hit light
                return ray.firstHitShape.lightColor
            else:
                # intersection point where object was hit
                ray.d = ray.d / np.linalg.norm(ray.d)
                intersPoint = ray.o + ray.d*ray.t

                if isinstance(ray.firstHitShape, Triangle)  :
                    v1v2 = ray.firstHitShape.v2 - ray.firstHitShape.v1
                    v1v3 = ray.firstHitShape.v3 - ray.firstHitShape.v1
                    intersectionNormal = np.cross(v1v3 / np.linalg.norm(v1v3), v1v2 / np.linalg.norm(v1v2))
                else :
                    # only for spheres
                    intersectionNormal = intersPoint

                # normalize normal vector
                intersectionNormal = intersectionNormal / np.linalg.norm(intersectionNormal)

                # sample
                brdfColor,brdfIntens = self.BRDFSampling(intersPoint, ray, scene, intersectionNormal)
                #brdfIntens = 0
                lsaColor, lsaIntens = self.LightSourceAreaSampling(intersPoint, ray, scene, intersectionNormal)


                # combine light colors of both sampling strategies
                lightColorsArray = np.array([
                    brdfColor,
                    lsaColor
                ])
                lightIntensitiesArray = np.array([
                    brdfIntens,
                    lsaIntens
                ])

                finalColor = self.weighColorByIntensity(lightColorsArray, lightIntensitiesArray)

                #average intensities for compression later
                finalIntensity = np.sum(lightIntensitiesArray) / len(lightIntensitiesArray)

                # interact light with object
                return ray.firstHitShape.color * finalColor * self.compressLightIntesity(finalIntensity)

        # no intersection so we stare into the deep void
        return [0.25,0.25,0.25]


    # for Triangles only
    def LightSourceAreaSampling(self, intersectionPoint, ray, scene, intersectionNormal):
        sumArea = 0.0
        areas = np.zeros(len(scene.lights),float)
        u2d=np.random.uniform(0.0,1.0,2)
        u1d=np.random.uniform()

        aquiredLightSum=0
        aquiredLight=0
        aquiredLightsCount=0
        aquiredLightsColor  = np.zeros((MISIntegrator.sampleCount, 3))
        aquiredLightsIntensity  = np.zeros(MISIntegrator.sampleCount)

        for sampleNr in range(MISIntegrator.sampleCount):
            shapeSet = []
            for n in range (len(scene.lights)) :

                #if coherent lightsource
                if scene.lights[n].bigSourceNumber>0:
                    if scene.lights[n].bigSourceNumber==1:
                        a = TriangleLight.TriangleArea(scene.lights[n-1])
                        areas[n-1]=a
                        sumArea += a
                        shapeSet.append(scene.lights[n-1])
                    a = TriangleLight.TriangleArea(scene.lights[n])
                    areas[n]=a
                    sumArea+=a
                    shapeSet.append(scene.lights[n])
                    continue

                light=scene.lights[n]
                sample=TriangleLight.TriangleSampling(light,u2d[0],u2d[1])
                #if self.VisibilityTest(intersectionPoint,sample,scene):
                    #aquiredLightsIntensity[n]=light.lightIntensity
                    #aquiredLightsColor[n]=light.lightColor
                    #aquiredLight=self.LightPower(light,TriangleLight.TriangleArea(light))
                    #aquiredLightsCount=+1
                #aquiredLightSum+=aquiredLight


            #shapeSet sampling
            sn = MISIntegrator.Distribution1Dcs(self,areas,len(areas),u1d)
            #sampled point on sampled shape
            pt = TriangleLight.TriangleSampling(scene.lights[sn],u2d[0],u2d[1])

            r = Ray(intersectionPoint,(pt-intersectionPoint)/np.linalg.norm(pt-intersectionPoint))

            lightSource=scene.lights[sn]

            if scene.intersect(r,shapeSet):
                org=r.o+r.d*r.t
                lightSource=r.firstHitShape

            org = pt
            dir = (org-intersectionPoint) / np.linalg.norm(org-intersectionPoint)

            self.LightSourceAreaSamplingPdf=self.LightSourceSetPdf(intersectionPoint,dir,shapeSet,areas,sumArea,scene)
            if self.LightSourceAreaSamplingPdf == 0:
                return [0,0,0],0

            if not self.VisibilityTest(org,intersectionPoint,scene):
               return [0,0,0],0

            aquiredLightsIntensity[aquiredLightsCount] = lightSource.lightIntensity / self.LightSourceAreaSamplingPdf
            aquiredLightsColor[aquiredLightsCount] = lightSource.lightColor
            aquiredLight = aquiredLightsIntensity[aquiredLightsCount] * sumArea * np.pi
            aquiredLightSum += aquiredLight
            aquiredLightsCount += 1
        ############################################################## Calculate Light

        combinedLightColor = np.zeros(3)

        # avoid / 0 when no light was aquired
        if aquiredLightSum > 0:
            #
            # calculate pixel color
            #

            # first calculate the color of the light hitting the shape
            # light that is more intense has more weight in the resulting color

            for n in range(aquiredLightsCount):
                combinedLightColor += aquiredLightsColor[n] * (aquiredLightsIntensity[n] / aquiredLightSum)

            # should not be necessary
            combinedLightColor = util.clipColor(combinedLightColor)

            # normalize light
            aquiredLightSum /= MISIntegrator.sampleCount


        # combine light color and object color + make it as bright as light that falls in
        # because we calculate the light value over an area we have to divide by area of hemisphere (2*pi)

        # because we can have tiny light sources and huge ones like the sun we need to
        # compress the dynamic range so a pc screen can still show the difference between
        # sunlight and a canle
        # log attenuates very high values and increases very small ones to an extent
        # small values are between 0-1 (+1 because log is only defined starting at 1)

        # dynamic compression can be adjusted by dividing factor. /2 means that all log(light) over 2 are the
        # brightest
        #value=ray.firstHitShape.color * combinedLightColor * (np.log((aquiredLightSum / 2 * np.pi) + 1) / 0.1)
        #print(value)

        return combinedLightColor, aquiredLightSum


    def Distance(self,p1,p2):
        return np.linalg.norm(p1-p2)

    def VisibilityTest(self,p1,p2,scene):
        dist=self.Distance(p1,p2)
        r=Ray(p1,(p2-p1)/dist)
        r.t=dist*1.001
        scene.intersectObjects(r)
        if r.t<dist:
            return False
        else:
            return True


    def LightPower(self,light,area):
        return light.lightIntensity*area*np.pi


    def LightSourcePdf(self,p,wi,light,scene):
        r=Ray(p,wi)
        if not scene.intersect(r,[light]):
            return 0.0
        intersecP=r.o+r.d*r.t

        lNormal=TriangleLight.TriangleNormal(light)
        #convert light sample weight to solid angle measure
        pdf = (self.Distance(p,intersecP)*self.Distance(p,intersecP)) / (np.abs(np.dot(lNormal,-wi))*TriangleLight.TriangleArea(light))
        return pdf



    def LightSourceSetPdf(self,p,wi,shapeSet,areas,sumArea,scene):
        pdf=0.0
        for i in range (len(shapeSet)):
            pdf += areas[i] * self.LightSourcePdf(p,wi,shapeSet[i],scene)
        return pdf / sumArea



    def UniformSampleHemisphere(self,u1,u2):
        z=u1
        r=np.sqrt(0 if (1.0-z*z)<0 else 1.0-z*z)
        phi = 2*np.pi*u2
        x = r*np.cos(phi)
        y = r*np.sin(phi)

        return np.array([x,y,z])


    def Distribution1Dcs(self,f,n,u):
        count = n

        cdf = np.zeros(n+1,float)

        for i in range(1,count+1):
            cdf[i]=cdf[i-1]+f[i-1] / n

        funcInt = cdf[count]
        for i in range(1,n+1):
            cdf[i] /= funcInt

        ptr=MISIntegrator.find_ge(self,cdf,u)
        dist=cdf[ptr]-u

        return int(cdf[ptr]-dist)

    #binary search
    def find_ge(self, a, x):
        'Find leftmost item greater than or equal to x'
        i = bisect_left(a, x)
        if i != len(a):
            return i
        raise ValueError



    def BRDFSampling(self, intersPoint, ray, scene, intersectionNormal):
        ############################################################## Prepare
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

        # Calculate matrix that rotates from the default hemisphere normal
        # to the intersection normal
        intersectionNormal[0] += 0.001
        intersectionNormal[1] += 0.001
        intersectionNormal[2] += 0.001

        sampleRoatationMatrix = self.rotation_matrix(np.cross(MISIntegrator.defaultHemisphereNormal, intersectionNormal),
                                            np.dot(MISIntegrator.defaultHemisphereNormal, intersectionNormal) * np.pi)


        # phi and theta how the camera looks at our intersection point
        camera_theta, camera_phi = self.VectorToAngles(ray.o - intersPoint)



        # precalculate theta cdfs for all light sources
        angleSteps = 20
        thetaCDFs = np.zeros((len(scene.lights),angleSteps))
        phiCDFs = np.zeros((len(scene.lights),angleSteps))

        for lightSourceNr in range(len(scene.lights)):
            pointOnLight = scene.lights[lightSourceNr].v1 * 0.33 + \
                           scene.lights[lightSourceNr].v2 * 0.33 + \
                           scene.lights[lightSourceNr].v3 * 0.33
            lightTheta, lightPhi = self.VectorToAngles(intersPoint - pointOnLight)

            # calc theta cdf
            for currAngleStep in range(angleSteps):
                # -pi to pi
                theta = -np.pi + (((2*np.pi) / angleSteps-1) * currAngleStep) + 0.01
                #print(theta)

                thetaCDFs[lightSourceNr][currAngleStep] = \
                    self.brdf.lookupValue(theta, lightPhi, camera_theta, camera_phi)[0]

            # calc phi cdf
            for currAngleStep in range(angleSteps):
                # 0 to 2*pi
                phi = ((2*np.pi) / angleSteps-1) * currAngleStep

                phiCDFs[lightSourceNr][currAngleStep] = \
                    self.brdf.lookupValue(lightTheta, phi, camera_theta, camera_phi)[0]

            thetaCDFs[lightSourceNr] = self.UnnormalizedFuncToCummulatedFunc(thetaCDFs[lightSourceNr])
            phiCDFs[lightSourceNr] = self.UnnormalizedFuncToCummulatedFunc(phiCDFs[lightSourceNr])

        """
        fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal', )
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.grid(False)
        for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
            for t in a.get_ticklines() + a.get_ticklabels():
                t.set_visible(False)
            a.line.set_visible(False)
            a.pane.set_visible(False)
        ax.view_init(elev=90, azim=0)

        omegasr3 = np.zeros((MISIntegrator.sampleCount, 3))

        for n in range(MISIntegrator.sampleCount):
            selectedLightIndex = int(np.round(np.random.random() * (len(scene.lights) - 1)))
            probabilityTheta = self.sampleOnCDF(thetaCDFs[selectedLightIndex])
            probabilityPhi = self.sampleOnCDF(phiCDFs[selectedLightIndex])

            theta = np.arccos(probabilityTheta)
            phi = (probabilityPhi * 2 - 1) * np.pi

            #theta = np.arccos(np.random.random())
            #phi = (np.random.random() * 2 - 1) * np.pi
            print("theta: ", theta, " phi:", phi)

            # map onto sphere
            # we get a point on the unit sphere that is oriented along the positive x axis
            lightSenseRaySecondPoint = self.twoAnglesTo3DPoint(theta, phi)
            omegasr3[n] = lightSenseRaySecondPoint # np.dot(roateMatrix, omegasr3[n])

        soa = np.zeros((2, 6))

        #soa[0, :] = [0, 0, 0, defaultNormal[0], defaultNormal[1], defaultNormal[2] * -1]
        #soa[1, :] = [0, 0, 0, alteredNormal[0], alteredNormal[1], alteredNormal[2] * -1]

        X, Y, Z, U, V, W = zip(*soa)

#        ax.quiver(X, Y, Z, U, V, W, color=[[0, 0, 1], [0, 1, 0]], pivot="tail", length=0.9)

        ax.scatter(omegasr3[:, 0], omegasr3[:, 1], omegasr3[:, 2] * -1)

        ax.set_xlabel('X........................')
        ax.set_ylabel('Y........................')
        ax.set_zlabel('Z........................')

        plt.show()
        return ([0,0,0],0)
        """

        # integrate over sphere using monte carlo
        for sampleNr in range(MISIntegrator.sampleCount):
            ############################################################## Sample Rays
            lightSenseRay = Ray(intersPoint)

            #
            # sample generation
            #
            # select a random light to decide which two cdfs to use
            selectedLightIndex = int(np.round(np.random.random() * (len(scene.lights) - 1)))


            # generate direction of light sense ray shot away from the hemisphere


            # generate theta and phi
            # to avoid points clustering at the top we use cos^-1 to convert angles from [0,1) to rad
            # for phi its sufficient to just map from [0,1] to -pi to pi
            probabilityTheta = self.sampleOnCDF(thetaCDFs[selectedLightIndex])
            #probabilityPhi = self.sampleOnCDF(phiCDFs[selectedLightIndex])
            theta = np.arccos(probabilityTheta)
            #phi = (probabilityPhi * 2 - 1) * np.pi

            #theta = np.arccos(np.random.random())
            phi = (np.random.random() * 2 - 1) * np.pi

            # map onto sphere
            # we get a point on the unit sphere that is oriented along the positive x axis
            lightSenseRaySecondPoint = self.twoAnglesTo3DPoint(theta, phi)
            #print(lightSenseRaySecondPoint)

            # but because we need a sphere that is oriented along the intersection normal
            # we rotate the point with the precalculated sample rotation matrix
            lightSenseRaySecondPoint = np.dot(sampleRoatationMatrix, lightSenseRaySecondPoint)

            # to get direction for ray we aquire the vector from the intersection point to our adjusted point on
            # the sphere
            #pointOnLight = scene.lights[selectedLightIndex].v1 * 0.33 + \
            #               scene.lights[selectedLightIndex].v2 * 0.33 + \
            #               scene.lights[selectedLightIndex].v3 * 0.33
            #lightSenseRay.d =  pointOnLight - intersPoint#-lightSenseRaySecondPoint
            #lightSenseRay.d /= np.linalg.norm(lightSenseRay.d)
            lightSenseRay.d = lightSenseRaySecondPoint
            #print(np.linalg.norm(lightSenseRay.d - lightSenseRaySecondPoint))

            # send ray on its way
            if scene.intersectLights(lightSenseRay) :
                aquiredLight = lightSenseRay.firstHitShape.lightIntensity

                # weigh light intensity by probability that this direction could be generated
                #aquiredLight /= np.maximum(probabilityTheta,0.001)
                #print(probabilityTheta)

                aquiredLightSum += aquiredLight

                aquiredLightsIntensity[aquiredLightsCount] = aquiredLight
                aquiredLightsColor[aquiredLightsCount] = lightSenseRay.firstHitShape.lightColor
                aquiredLightsCount += 1

        ############################################################## Calculate Light
        combinedLightColor = np.zeros(3)

        # avoid / 0 when no light was aquired
        if aquiredLightSum > 0 :
            #
            # calculate pixel color
            #

            # first calculate the color of the light hitting the shape
            combinedLightColor = \
                self.weighColorByIntensity(aquiredLightsIntensity, aquiredLightsColor, aquiredLightSum)

            # normalize lightIntensity
            aquiredLightSum /= MISIntegrator.sampleCount

            #if ray.firstHitShape.tri:
            #    for n in range(len(debugRayList)):
            #        debugRayList[n].print2(n)
        #else:
            """
            if ray.firstHitShape.tri:
                for n in range(len(debugRayList)):
                    debugRayList[n].print2(n)
            """
        #    return [0,1,0]

        # combine light color and object color + make it as bright as light that falls in
        # because we calculate the light value over an area we have to divide by area of hemisphere (2*pi)

        # because we can have tiny light sources and huge ones like the sun we need to
        # compress the dynamic range so a pc screen can still show the difference between
        # sunlight and a canle
        # log attenuates very high values and increases very small ones to an extent
        # small values are between 0-1 (+1 because log is only defined starting at 1)

        # dynamic compression can be adjusted by dividing factor. /2 means that all log(light) over 2 are the
        # brightest

        return combinedLightColor, aquiredLightSum / (2*np.pi)

    def VectorToAngles(self, vec):
        """
        Calculates theta, phi from direction of given vector
        :param normal:
        :param incomingVec:
        :return:
        """
        vec /= np.linalg.norm(vec)
        theta = np.arccos(vec[2])
        phi = np.arccos(np.minimum(np.maximum(vec[0] / np.maximum(np.sin(theta),0.0001), -1), 1))
        return theta, phi



    def integrateArray0To1(self, array):
        """
        Sees the array as Y of a function going from 0 to 1
        :param array:
        :return: Area of function
        """
        sliceSize = 1 / len(array)
        value = 0
        for n in range(len(array)):
            value += array[n] * sliceSize
        return value

    def UnnormalizedFuncToCummulatedFunc(self, unnormalizedSamples):
        """
        Takes an array of samples from a function, sees it as going from 0 to 1 with  len(unnormalizedSamples) values
        in between,  normalizes it so its area is 1 and calculates the cummulated function of it
        :param unnormalizedSamples:
        :return:
        """
        unnormalizedSamples = np.clip(unnormalizedSamples, a_min=0, a_max=999999)
        area = self.integrateArray0To1(unnormalizedSamples)
       #  unnormalizedSamples /= area
        return np.cumsum(unnormalizedSamples / (area*len(unnormalizedSamples)))

    def sampleOnCDF(self, cdf):
        """
        Uses uniform samples and samples on the given cdf
        :param cdf:
        :return:
        """
        sample = np.random.uniform(0,1)
        # return (1.0 / len(cdf)) * np.searchsorted(cdf, [sample])

        for i in range(len(cdf)):
            if cdf[i] > sample:
                if i == 0 or cdf[i] - sample < cdf[i-1] - sample:
                    return (1.0/len(cdf)) * i
                else:
                    return (1.0 / len(cdf)) * (i-1)
        # when no fitting spot was found
        return 1




    def compressLightIntesity(self, intensity):
        if intensity < 0:
            intensity = 0
        return intensity# np.log(intensity+1)/2

    # light that is more intense has more weight in the resulting color
    def weighColorByIntensity(self, colorArray, intensityArray, intensitySum=None):
        if len(colorArray) != len(intensityArray):
            raise Exception
        finalColor = np.zeros(3)
        if intensitySum is None:
            intensitySum = np.sum(intensityArray)
        for i in range(len(colorArray)):
            finalColor += colorArray[i] * (intensityArray[i] / intensitySum)
        return finalColor


    """
    theta, phi angles in radiant
    Returns vector from coordinate origin to point on unit sphere where both angles would put it
    """
    def twoAnglesTo3DPoint(self, theta, phi):
        r3 = np.zeros(3)
        r3[0] = np.sin(theta) * np.cos(phi)
        r3[1] = np.sin(theta) * np.sin(phi)
        r3[2] = np.cos(theta)
        return r3


    """
    Renerates rotation matrix from given axis and angle
    """
    def rotation_matrix(self, axis, theta):
        # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / np.maximum(np.sqrt(np.dot(axis, axis)), 0.00001)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
