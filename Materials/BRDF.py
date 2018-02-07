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
