import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# raytracers coordinate system can be seen in camera.py
# plotters coordinate system must be rotated and Z negated to match the raytracers one



# last triangle is always red so you can see which one you are editing when true
# color you specified otherwise
lastTriangleRed = True


# one row is one triangle
# row format is [ v1,  v2, v3, color ]
# coordinates in XYZ form
# vertex order is CCW (not important for this but later for raytracer)
polyArray = [
    [[5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [5.0, -5.0, 12.0], [1,1,1]],        #floor

    [[5.0, 5.0, 0.0], [5.0, 5.0, 12.0], [5.0, -5.0, 12.0], [1,1,1]],        #floor

    [[5.0, -5.0, 0.0], [5.0, -5.0, 12.0], [-5.0, -5.0, 0.0], [1,0,0]],      #left wall

    [[5.0, -5.0, 12.0], [-5.0, -5.0, 12.0], [-5.0, -5.0, 0.0], [1,0,0]],    #left wall

    [[5.0, 5.0, 0.0], [5.0, 5.0, 12.0], [-5.0, 5.0, 0.0], [0, 1, 0]],       #right wall

    [[5.0, 5.0, 12.0], [-5.0, 5.0, 12.0], [-5.0, 5.0, 0.0], [0, 1, 0]],     #right wall

    [[5.0, -5.0, 12.0], [5.0, 5.0, 12.0], [-5.0, 5.0, 12.0], [1, 1, 1]],    #back wall

    [[5.0, -5.0, 12.0], [-5.0, 5.0, 12.0], [-5.0, -5.0, 12.0], [1, 1, 1]],  #back wall

    [[-5.0, -5.0, 0.0], [-5.0, 5.0, 0.0], [-5.0, -5.0, 12.0], [1,1,1]],     #ceiling

    [[-5.0, 5.0, 0.0], [-5.0, 5.0, 12.0], [-5.0, -5.0, 12.0], [1,1,1]],     #ceiling

]






fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal',)
ax.set_xlim( -10.1, 10.1)
ax.set_ylim( -10.1, 10.1)
ax.set_zlim( -10.1, 10.1)
ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines()+a.get_ticklabels():
        t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)
# roughly adjust to raytracers camera position
ax.view_init(elev=90, azim=0)

"""
#plot points
ax.scatter(2,0,3, c=[[1,0,0]])
ax.scatter(-2,0,3, c=[[1,0,0]])
ax.scatter(0,0,0, c=[[1,0,0]])
"""



compatiblePolyArray = np.zeros((len(polyArray), 3, 3))
for n in range(len(polyArray)) :
    compatiblePolyArray[n, 0, :] = polyArray[n][0]
    compatiblePolyArray[n, 1, :] = polyArray[n][1]
    compatiblePolyArray[n, 2, :] = polyArray[n][2]
    compatiblePolyArray[n][0][2] *= -1
    compatiblePolyArray[n][1][2] *= -1
    compatiblePolyArray[n][2][2] *= -1


# create color array
colorArray = np.zeros((len(compatiblePolyArray), 3))
for n in range(len(compatiblePolyArray)-1) :
    if lastTriangleRed :
        colorArray[n, :] = [0.75,0.75,0.75]
    else :
        colorArray[n, :] = polyArray[n][3]

if lastTriangleRed:
    colorArray[len(compatiblePolyArray)-1, :] = [1,0,0]

# plot triangles
ax.add_collection3d(Poly3DCollection(compatiblePolyArray, color=colorArray))


# plot coordinate arrows
# x = red, y = green, z = blue
soa = np.zeros((3 + len(compatiblePolyArray),6))

soa[0,:] = [0, 0, 0,  1, 0, 0]
soa[1,:] = [0, 0, 0,  0, 1, 0]
soa[2,:] = [0, 0, 0,  0, 0, -1]


uvw = np.array([0.33,0.33,0.33])

# add triangle face arrow
for n in range(len(compatiblePolyArray)):
    arrowStart = np.array(compatiblePolyArray[n][0]) * uvw[0] +  \
                 np.array(compatiblePolyArray[n][1]) * uvw[1] + \
                 np.array(compatiblePolyArray[n][2]) * uvw[2]

    arrowDir = np.cross(np.array(compatiblePolyArray[n][1]) -  np.array(compatiblePolyArray[n][0]),
                        np.array(compatiblePolyArray[n][2]) -  np.array(compatiblePolyArray[n][0]),
                            )
    soa[3+n, :] = [
            arrowStart[0],
            arrowStart[1],
            arrowStart[2],
            arrowDir[0],
            arrowDir[1],
            arrowDir[2]
        ]

X, Y, Z, U, V, W = zip(*soa)
ax.quiver(X, Y, Z, U, V, W, color=[[1,0,1],[0,1,0],[0,0,1],[0,0,0]], pivot="tail", length=1)


ax.set_xlabel('X........................')
ax.set_ylabel('Y........................')
ax.set_zlabel('Z........................')
plt.show()