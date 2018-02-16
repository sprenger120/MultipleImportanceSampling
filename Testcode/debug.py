import numpy as np

def VectorToAngles(vec):
    """
    Calculates theta, phi from direction of given vector
    :param normal:
    :param incomingVec:
    :return:
    """
    vec /= np.linalg.norm(vec)
    theta = np.arccos(vec[2])
    a = vec[0] / np.sin(theta)
    phi = np.arccos(np.minimum(np.maximum(vec[0] / np.sin(theta),-1),1))
    return theta, phi







if __name__ == "__main__":
    interPoint = np.array([-5.0, 0, 0.24])
    o = np.array([0, 0, -10])
    camera_theta, camera_phi = VectorToAngles(interPoint - o)

    print ("c_t: ", camera_theta, "c_p: ", camera_phi)