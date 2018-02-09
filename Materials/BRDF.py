import os
import struct
import numpy as np

class BRDF:
    BRDF_SAMPLING_RES_THETA_H =  90
    BRDF_SAMPLING_RES_THETA_D =  90
    BRDF_SAMPLING_RES_PHI_D  =  360

    # define RED_SCALE (1.0/1500.0)
    # define GREEN_SCALE (1.15/1500.0)
    # define BLUE_SCALE (1.66/1500.0)


    def __init__(self):
        self.data = 0
        return


    def normalizeVector(self, vec):
        return vec / np.linalg.norm(vec)

    def rotateVector(self, vector, axis, angle):
        cosAngle = np.cos(angle)
        out = vector*cosAngle
        temp = np.dot(axis,vector) * (1-cosAngle)
        out += axis * temp
        out += np.cross(axis, vector) * np.sin(angle)
        return out


    def standardCoordsToHalfDiffCoords(self, theta_in, phi_in, theta_out, phi_out):
        """
        0: theta_half
        1: phi_half
        2: theta_diff
        3: phi_diff
        """
        output = np.zeros(4)

        # compute in vector
        projInVec = np.sin(theta_in)
        vecIn = np.array(
            [projInVec * np.cos(phi_in),
             projInVec * np.sin(phi_in),
             np.cos(theta_in)
             ])
        vecIn = self.normalizeVector(vecIn)

        # compute out vector
        projOutVec = np.sin(theta_out)
        vecOut = np.array(
            [projInVec * np.cos(phi_out),
             projInVec * np.sin(phi_out),
             np.cos(theta_out)
             ])
        vecOut = self.normalizeVector(vecOut)

        # compute halfway vector
        half = (vecIn + vecOut) / 2.0
        half = self.normalizeVector(half)

        # compute  theta_half, phi_half
        output[0] = np.arccos(half[2])
        output[1] = np.arctan2(half[0], half[1])

        




    def lookupValue(self, theta_in, phi_in, theta_out, phi_out):
        out = np.zeros(3)



        return out


    def loadBRDF(self, fileName):
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

        if n != BRDF.BRDF_SAMPLING_RES_PHI_D * BRDF.BRDF_SAMPLING_RES_THETA_D * BRDF.BRDF_SAMPLING_RES_THETA_H / 2:
            raise Exception()

        self.data = struct.unpack_from(str(n*3) + 'd',content,12)
        print("Loaded brdf: ", fileName)
        return




if __name__ == "__main__":
    print(os.getcwd())
    brdf = BRDF()
    brdf.loadBRDF(
        os.path.join(os.getcwd(), "brdf_data","alum-bronze.binary"))
