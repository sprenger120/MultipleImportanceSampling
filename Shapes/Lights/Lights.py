from Shapes.Lights.LightBase import LightBase
from Shapes.sphere import Sphere
from Shapes.Triangle import Triangle


class SphereLight (LightBase, Sphere) :

    def __init__(self, pos, r, lightColor, lightIntensity) :
        LightBase.__init__(self, lightColor, lightIntensity)
        Sphere.__init__(self, pos, r, lightColor)
        return

class TriangleLight(LightBase, Triangle) :

    def __init__(self, v1, v2,v3, lightColor, lightIntensity) :
        LightBase.__init__(self, lightColor, lightIntensity)
        Triangle.__init__(self, v1,v2,v3, lightColor)
        return