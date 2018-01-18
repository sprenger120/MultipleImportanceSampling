#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:49:01 2017

@author: lessig
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

###############################################################################
# function definitions

###############################################################################
def s2tor3( omega) :
    r3 = np.zeros( 3)
    r3[0] = np.sin( omega[0]) * np.cos( omega[1])
    r3[1] = np.sin( omega[0]) * np.sin( omega[1])
    r3[2] = np.cos( omega[0])
    return r3

def r3Tos2(r3) :
    # omegas[0] = theta
    # omegas[1] = phi

    theta = np.arccos(r3[2])

    #  r3[0] = np.sin( theta ) * np.cos( phi)
    phi = np.arccos(r3[0] / np.sin(theta))

    return [theta, phi]

###############################################################################
def s2tor3array( omega) :
    r3 = np.zeros( [omega.shape[0], 3])
    r3[:,0] = np.sin( omega[:,0]) * np.cos( omega[:,1])
    r3[:,1] = np.sin( omega[:,0]) * np.sin( omega[:,1])
    r3[:,2] = np.cos( omega[:,0])
    #r3 += [1,1,1]
    return r3

###############################################################################
def ell( omega) :
    return np.max( [0.0,  np.cos(omega[0])])



###############################################################################
def getUniformPointS2() :

    # generate random point in [0,1]^2
    omega =  np.zeros(2)


    #theta
    omega[0] = (np.random.random() * (2) - 1) * (np.pi / 2)
    #phi
    omega[1] = (np.random.random() * 2 -1) * (np.pi)

    # transform points to globe

    return omega


def rotation_matrix_numpy(axis, theta):
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    mat = np.eye(3, 3)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)

    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

###############################################################################
# integration

defaultNormal = np.array([0,0,1])
alteredNormal = [0,0.75,0.75]
alteredNormal = alteredNormal / np.linalg.norm(alteredNormal)

print(alteredNormal)
# angles = r3Tos2

a = alteredNormal

#angleX = 0.5
#angleY = 0
#angleZ = 0

roateMatrix = rotation_matrix_numpy(np.cross(alteredNormal,defaultNormal), np.dot(defaultNormal, alteredNormal))
#roateMatrixY = rotation_matrix_numpy([0,1,0],angleY * np.pi)
#roateMatrixZ = rotation_matrix_numpy([0,0,1],angleZ * np.pi)


#roateMatrix = np.matmul(np.matmul(roateMatrixX,roateMatrixY), roateMatrixZ)


N = 512
omegas = np.zeros([N, 2])
for n in range(N):
    omegas[n,:] = getUniformPointS2()



###############################################################################
# plot points

"""
omegas [0] = theta
omegas[1] = phi

"""

fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal',)
ax.set_xlim( -1.1, 1.1)
ax.set_ylim( -1.1, 1.1)
ax.set_zlim( -1.1, 1.1)
ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines()+a.get_ticklabels():
        t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)

omegasr3 = s2tor3array( omegas)


for n in range(N):
    omegasr3[n] = np.dot(roateMatrix, omegasr3[n])




soa = np.zeros((2,6))

soa[0,:] = [0, 0, 0,  defaultNormal[0], defaultNormal[1], defaultNormal[2]]
soa[1,:] = [0, 0, 0,  alteredNormal[0], alteredNormal[1], alteredNormal[2]]

X, Y, Z, U, V, W = zip(*soa)

ax.quiver(X, Y, Z, U, V, W, color=[[0,0,1],[0,1,0]], pivot="tail")


ax.scatter(0, 0, 0, c=[[1,0,0]])
ax.scatter( omegasr3[:,0], omegasr3[:,1], omegasr3[:,2])

ax.set_xlabel('X........................')
ax.set_ylabel('Y........................')
ax.set_zlabel('Z........................')

plt.show()
