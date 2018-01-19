


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection





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

"""
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


"""

"""
d  [-0.06488996  0.30776382  0.94924745]  o  [0, 0, -10]  t  11.2173488997
normal  [ 3. -3.  9.]
"""

# draw vectors
t = 50
d = np.array([-0.06488996,  0.30776382 , 0.94924745])
o = np.array([0, 0, -10])
d *= t
p = o + d

normal = [ 3. ,-3.,  9.]
soa = np.zeros((1,6))
soa[0,:] = [o[0], o[1], o[2],  d[0], d[1], d[2]]
#soa[1,:] = [0, 0, 0,  normal[0], normal[1], normal[2]]

X, Y, Z, U, V, W = zip(*soa)

ax.quiver(X, Y, Z, U, V, W, color=[[0,0,1],[0,1,0]], pivot="tail", length=t)

ax.scatter(p[0],p[1],p[2], c=[[1,0,0]])

ax.add_collection3d(Poly3DCollection([[np.array([0.0,2.0,4.0]),np.array([3.0,2.0,3.0]),np.array([3.0,5.0,4.0])]]))

ax.set_xlabel('X........................')
ax.set_ylabel('Y........................')
ax.set_zlabel('Z........................')
plt.show()