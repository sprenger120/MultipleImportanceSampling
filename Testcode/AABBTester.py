


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from Shapes.Triangle import Triangle
from Shapes.sphere import Sphere
from Shapes.shape import Shape
from Scene.Octree import BoundingVolume,Octree,OctreeNode


def plotSphere(X,Y,Z, r):
    global ax
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)

    x = np.outer(np.sin(u), np.sin(v)) * r + X
    y = np.outer(np.sin(u), np.cos(v)) * r + Y
    z = np.outer(np.cos(u), np.ones_like(v)) * r +Z

    ax.plot_wireframe(x, y, z, color=[0.5,0.5,0.5])
    return

def drawAABB(sh):
    global ax
    global plt
    ax.scatter(sh.BBv1[0], sh.BBv1[1], -sh.BBv1[2], c=[[0, 0, 0]])
    ax.scatter(sh.BBv2[0], sh.BBv2[1], -sh.BBv2[2], c=[[0.25, 0.25, 0.25]])
    ax.scatter(sh.BBv3[0], sh.BBv3[1], -sh.BBv3[2], c=[[0.5, 0.5, 0.5]])
    ax.scatter(sh.BBv4[0], sh.BBv4[1], -sh.BBv4[2], c=[[0.75, 0.75, 0.75]])

    ax.scatter(sh.BBv5[0], sh.BBv5[1], -sh.BBv5[2], c=[[0.93, 0.41, 0.41]])
    ax.scatter(sh.BBv6[0], sh.BBv6[1], -sh.BBv6[2], c=[[0.97, 0.48, 0.48]])
    ax.scatter(sh.BBv7[0], sh.BBv7[1], -sh.BBv7[2], c=[[0.99, 0.66, 0.66]])
    ax.scatter(sh.BBv8[0], sh.BBv8[1], -sh.BBv8[2], c=[[0.94, 0.75, 0.75]])

    v1 = np.array([sh.BBv1[0],sh.BBv1[1],-sh.BBv1[2]])
    v2 = np.array([sh.BBv2[0],sh.BBv2[1],-sh.BBv2[2]])
    v3 = np.array([sh.BBv3[0],sh.BBv3[1],-sh.BBv3[2]])
    v4 = np.array([sh.BBv4[0],sh.BBv4[1],-sh.BBv4[2]])
    v5 = np.array([sh.BBv5[0],sh.BBv5[1],-sh.BBv5[2]])
    v6 = np.array([sh.BBv6[0],sh.BBv6[1],-sh.BBv6[2]])
    v7 = np.array([sh.BBv7[0],sh.BBv7[1],-sh.BBv7[2]])
    v8 = np.array([sh.BBv8[0],sh.BBv8[1],-sh.BBv8[2]])

    ax.plot([v1[0],v2[0]],[v1[1],v2[1]], [v1[2],v2[2]])
    ax.plot([v2[0],v3[0]],[v2[1],v3[1]], [v2[2],v3[2]])
    ax.plot([v3[0],v4[0]],[v3[1],v4[1]], [v3[2],v4[2]])
    ax.plot([v4[0],v1[0]],[v4[1],v1[1]], [v4[2],v1[2]])
    ax.plot([v1[0],v5[0]],[v1[1],v5[1]], [v1[2],v5[2]])
    ax.plot([v2[0],v6[0]],[v2[1],v6[1]], [v2[2],v6[2]])
    ax.plot([v3[0],v7[0]],[v3[1],v7[1]], [v3[2],v7[2]])
    ax.plot([v4[0],v8[0]],[v4[1],v8[1]], [v4[2],v8[2]])
    ax.plot([v5[0],v6[0]],[v5[1],v6[1]], [v5[2],v6[2]])
    ax.plot([v6[0],v7[0]],[v6[1],v7[1]], [v6[2],v7[2]])
    ax.plot([v8[0],v5[0]],[v8[1],v5[1]], [v8[2],v5[2]])
    ax.plot([v7[0],v8[0]],[v7[1],v8[1]], [v7[2],v8[2]])

    return


def plotSphereFromShape(sph):
    plotSphere(sph.pos[0], sph.pos[1], -sph.pos[2], sph.r)
    return


fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal',)
ax.set_xlim( -3.1, 3.1)
ax.set_ylim( -3.1, 3.1)
ax.set_zlim( -3.1, 3.1)
ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines()+a.get_ticklabels():
        t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)
ax.view_init(elev=90, azim=0)





#aabb
OCTREE_COORDINATE_MAX = 10
rootBoundingVolume = BoundingVolume(single=True)
rootBoundingVolume.BBv2 = np.array([-OCTREE_COORDINATE_MAX,
                                         -OCTREE_COORDINATE_MAX,
                                         -OCTREE_COORDINATE_MAX])
rootBoundingVolume.BBv8 = np.array([OCTREE_COORDINATE_MAX,
                                         OCTREE_COORDINATE_MAX,
                                         OCTREE_COORDINATE_MAX])
rootBoundingVolume.finalizeAABB()
node = OctreeNode(rootBoundingVolume, [])
node.initializeOctants()


drawAABB(rootBoundingVolume)
drawAABB(node.octants[0].boundingVolume)

node.octants[0].initializeOctants()

drawAABB(node.octants[0].octants[0].boundingVolume)

node.octants[0].octants[0].initializeOctants()

drawAABB(node.octants[0].octants[0].octants[0].boundingVolume)


node.octants[0].octants[0].octants[0].initializeOctants()

drawAABB(node.octants[0].octants[0].octants[0].octants[0].boundingVolume)




#sphere
#s1 = Sphere(np.array([0, 0, 0]), 1, np.array([0,0,0]))
#s2 = Sphere(np.array([2, 2, 2]), 0.5, np.array([0,0,0]))

#plotSphereFromShape(s1)
#plotSphereFromShape(s2)
#drawAABB(s1)
#drawAABB(s2)




#triangle
t1 = Triangle([5.0, 2.0, 0.0], [5.0, 5.0, 10.0], [-5.0, 5.0, -10.0], [1, 1, 1])
t2 = Triangle([5.0, 3.0, 2.0], [5.0, 3.0, 4.0], [3.0, 3.0, 4.0], [1, 1, 1])

t = t1

polyT = [[t.v1,t.v2,t.v3]]

polyT[0][0][2] *= -1
polyT[0][1][2] *= -1
polyT[0][2][2] *= -1

#drawAABB(t)
# ax.add_collection3d(Poly3DCollection(polyT, color=[[0.5,0.5,0.5]]))




ax.set_xlabel('X........................')
ax.set_ylabel('Y........................')
ax.set_zlabel('Z........................')
plt.show()

