from Shapes.Lights.LightBase import LightBase
from Shapes.sphere import Sphere


class SphereLight (LightBase, Sphere) :

    def __init__(self, pos, r, lightColor, lightIntensity) :
        LightBase.__init__(self, lightColor, lightIntensity)
        Sphere.__init__(self, pos, r, lightColor)
        return
