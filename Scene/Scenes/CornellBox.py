from Scene.scene import Scene
from Shapes.Lights.Lights import TriangleLight
import numpy as np

class CornellBox(Scene):

    def __init__(self):
        Scene.__init__(self)
        return


    def importGeometry(self):
        # can be directly taken from fast triangle viewer
        # only contains triangles
        polyArray = [
            [[5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [5.0, -5.0, 10.0], [1, 1, 1]],  # floor

            [[5.0, 5.0, 0.0], [5.0, 5.0, 10.0], [5.0, -5.0, 10.0], [1, 1, 1]],  # floor

            [[5.0, -5.0, 0.0], [5.0, -5.0, 10.0], [-5.0, -5.0, 0.0], [1, 0, 0]],  # left wall

            [[5.0, -5.0, 10.0], [-5.0, -5.0, 10.0], [-5.0, -5.0, 0.0], [1, 0, 0]],  # left wall

            [[5.0, 5.0, 0.0], [5.0, 5.0, 10.0], [-5.0, 5.0, 0.0], [0, 1, 0]],  # right wall

            [[5.0, 5.0, 10.0], [-5.0, 5.0, 10.0], [-5.0, 5.0, 0.0], [0, 1, 0]],  # right wall

            [[5.0, -5.0, 10.0], [5.0, 5.0, 10.0], [-5.0, 5.0, 10.0], [1, 1, 1]],  # back wall

            [[5.0, -5.0, 10.0], [-5.0, 5.0, 10.0], [-5.0, -5.0, 10.0], [1, 1, 1]],  # back wall

            [[-5.0, -5.0, 0.0], [-5.0, 5.0, 0.0], [-5.0, -5.0, 10.0], [1, 1, 1]],  # ceiling

            [[-5.0, 5.0, 0.0], [-5.0, 5.0, 10.0], [-5.0, -5.0, 10.0], [1, 1, 1]],  # ceiling

            [[5.0, 1.0, 2.0], [5.0, 3.0, 2.0], [3.0, 3.0, 2.0], [1, 1, 1]],  # first block

            [[5.0, 1.0, 2.0], [3.0, 3.0, 2.0], [3.0, 1.0, 2.0], [1, 1, 1]],  # first block

            [[5.0, 3.0, 2.0], [5.0, 3.0, 4.0], [3.0, 3.0, 4.0], [1, 1, 1]],  # first block

            [[5.0, 3.0, 2.0], [3.0, 3.0, 4.0], [3.0, 3.0, 2.0], [1, 1, 1]],  # first block

            [[3.0, 1.0, 2.0], [3.0, 3.0, 2.0], [3.0, 3.0, 4.0], [1, 1, 1]],  # first block

            [[3.0, 1.0, 2.0], [3.0, 3.0, 4.0], [3.0, 1.0, 4.0], [1, 1, 1]],  # first block
        ]
        self.importPolyArray(polyArray)

        self.lights.append(
            TriangleLight(np.array([-5.0, -1.0, 3.0]), np.array([-5.0, -1.0, 4.0]), np.array([-5.0, 1.0, 3.0]),
                          [1, 1, 1], 1, 0)
        )

        self.lights.append(
            TriangleLight(np.array([-5.0, -1.0, 4.0]), np.array([-5.0, 1.0, 4.0]), np.array([-5.0, 1.0, 3.0]),
                          [1, 1, 1], 1, 1)
        )
        return