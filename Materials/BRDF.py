import os
import struct
import numpy as np

class BRDF:
    __BRDF_SAMPLING_RES_THETA_H =  90
    __BRDF_SAMPLING_RES_THETA_D =  90
    __BRDF_SAMPLING_RES_PHI_D  =  360

    __normal = np.array([0.0, 0.0, 1.0])
    __bi_normal = np.array([0.0, 1.0, 0.0])


    __RED_SCALE =  (1.0 / 1500.0)
    __GREEN_SCALE =  (1.15 / 1500.0)
    __BLUE_SCALE = (1.66 / 1500.0)


    def __init__(self, fileName):
        self.__data = 0
        self.__loadBRDF(fileName)
        return

    def __normalizeVector(self, vec):
        return vec / np.linalg.norm(vec)

    def __rotateVector(self, vector, axis, angle):
        cosAngle = np.cos(angle)
        out = vector*cosAngle
        temp = np.dot(axis,vector) * (1-cosAngle)
        out += axis * temp
        out += np.cross(axis, vector) * np.sin(angle)
        return out


    def __standardCoordsToHalfDiffCoords(self, theta_in, phi_in, theta_out, phi_out):
        theta_half = 0
        phi_half = 0
        theta_diff = 0
        phi_diff = 0
        # compute in vector
        projInVec = np.sin(theta_in)
        vecIn = np.array(
            [projInVec * np.cos(phi_in),
             projInVec * np.sin(phi_in),
             np.cos(theta_in)
             ])
        vecIn = self.__normalizeVector(vecIn)


        # compute out vector
        projOutVec = np.sin(theta_out)
        vecOut = np.array(
            [projInVec * np.cos(phi_out),
             projInVec * np.sin(phi_out),
             np.cos(theta_out)
             ])
        vecOut = self.__normalizeVector(vecOut)


        # compute halfway vector
        half = (vecIn + vecOut) / 2.0
        half = self.__normalizeVector(half)

        # compute  theta_half, phi_half
        theta_half = np.arccos(half[2])
        phi_half = np.arctan2(half[1], half[0])

        # compute diff vector
        temp = self.__rotateVector(vecIn, BRDF.__normal, -phi_half)
        diff = self.__rotateVector(temp, BRDF.__bi_normal, -theta_half)

        theta_diff = np.arccos(diff[2])
        phi_diff = np.arctan2(diff[1], diff[0])

        return (theta_half, phi_half, theta_diff, phi_diff)


    def __LookupPhiDiffIndex(self, phi_diff):
        # Because of reciprocity, the BRDF is unchanged under
        # phi_diff -> phi_diff + M_PI
        if phi_diff < 0.0:
            phi_diff += np.pi

        # In: phi_diff in [0..pi]
        # Out: tmp in [0.. 179]
        temp = np.floor(phi_diff / np.pi * BRDF.__BRDF_SAMPLING_RES_PHI_D / 2)
        if temp < 0:
            return 0
        if temp < (BRDF.__BRDF_SAMPLING_RES_PHI_D / 2) - 1:
            return np.int(temp)
        else:
            return np.int((BRDF.__BRDF_SAMPLING_RES_PHI_D / 2) - 1)


    # Lookup theta_diff index
    # In: [0..pi / 2]
    # Out: [0.. 89]
    def __LookupThetaDiffIndex(self, theta_diff):
        temp = np.floor(theta_diff / (np.pi / 2) * BRDF.__BRDF_SAMPLING_RES_THETA_D)
        if temp < 0:
            return 0
        if temp < BRDF.__BRDF_SAMPLING_RES_THETA_D -1:
            return np.int(temp)
        else:
            return BRDF.__BRDF_SAMPLING_RES_THETA_D - 1


    # Lookup theta_half index
    # This is a non - linear mapping
    # In: [0..pi / 2]
    # Out: [0.. 89]
    def __LookupThetaHalfIndex(self, theta_half):
        if theta_half <= 0.0:
            return 0
        theta_half_deg = (theta_half / (np.pi / 2)) * BRDF.__BRDF_SAMPLING_RES_THETA_H
        temp = np.int(np.sqrt(theta_half_deg * BRDF.__BRDF_SAMPLING_RES_THETA_H))
        if temp < 0:
            return 0
        if temp >= BRDF.__BRDF_SAMPLING_RES_THETA_H:
            return BRDF.__BRDF_SAMPLING_RES_THETA_H - 1
        return temp

    def lookupValue(self, theta_in, phi_in, theta_out, phi_out):
        out = np.zeros(3)
        theta_half, phi_half, theta_diff, phi_diff = \
        self.__standardCoordsToHalfDiffCoords(theta_in, phi_in, theta_out, phi_out)

        # Find index.
        # Note that phi_half is ignored, since isotropic BRDFs are assumed
        index = np.int(self.__LookupPhiDiffIndex(phi_diff) + \
                       self.__LookupThetaDiffIndex(theta_diff) * BRDF.__BRDF_SAMPLING_RES_PHI_D / 2 + \
                       self.__LookupThetaHalfIndex(theta_half) * BRDF.__BRDF_SAMPLING_RES_PHI_D / 2 * \
                       BRDF.__BRDF_SAMPLING_RES_THETA_D)

        offset = BRDF.__BRDF_SAMPLING_RES_THETA_H * BRDF.__BRDF_SAMPLING_RES_THETA_D * BRDF.__BRDF_SAMPLING_RES_PHI_D
        return np.array([
            self.__data[index] * BRDF.__RED_SCALE,
            self.__data[index + np.int(offset / 2)] * BRDF.__GREEN_SCALE,
            self.__data[index + np.int(offset)] * BRDF.__BLUE_SCALE])

    def __loadBRDF(self, fileName):
        file = open(fileName, "rb")
        # should be only 30mb big so maybe we are ok
        content = file.read()
        file.close()

        # read dimensions from file
        # haven't found out in what order theta_h theta_d and phi_d are inside the dimensions
        # sections so we have to reject any brdf file that doesn't behave like the demo parser code
        # wants it to be
        dim = struct.unpack_from('3i',content,0)
        n = dim[0]*dim[1]*dim[2]

        if n != BRDF.__BRDF_SAMPLING_RES_PHI_D * BRDF.__BRDF_SAMPLING_RES_THETA_D * BRDF.__BRDF_SAMPLING_RES_THETA_H / 2:
            raise Exception()

        self.__data = struct.unpack_from(str(n * 3) + 'd', content, 12)
        print("Loaded brdf: ", fileName)
        return
