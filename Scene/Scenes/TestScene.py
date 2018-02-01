from Scene.scene import Scene
from Shapes.Lights.Lights import TriangleLight, SphereLight
from Shapes.sphere import Sphere
import numpy as np

class TestScene(Scene):
    def __init__(self):
        Scene.__init__(self)
        return

    def importGeometry(self):
        self.objects.append(
            Sphere(np.array([0, 0.0, 0]), 0.2, [1, 1, 1])
        )

        self.objects.append(
            Sphere(np.array([1, 0.0, 0]), 0.2, [1, 0, 0])
        )

        self.objects.append(
            Sphere(np.array([0, 1.0, 0]), 0.2, [0, 1, 0])
        )

        self.objects.append(
            Sphere(np.array([-1, 0.0, 5]), 0.2, [0, 0, 1])
        )

        self.lights.append(
            SphereLight(np.array([0, 0, 3.0]), 0.5,  # position, radius
                        [1, 1, 1], 0.5)  # light color, light intensity
        )
        return