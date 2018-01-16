# a light is a shape that emits light, so when constructing a light just use this class as
# the base in combination with the shape you want to have

from abc import ABCMeta

class LightBase :
    __metaclass__ = ABCMeta

    def __init__(self, lightColor, lightIntensity) :
        self.lightColor = lightColor
        self.lightIntensity = lightIntensity
        return
