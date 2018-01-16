
from abc import ABCMeta
import util as util

"""
Base class for all materials
Materials define the objects behavior under light
"""


class Material :
    _metaclass__ = ABCMeta

    def __init__(self, color):
        if not util.isColor(color) :
            raise Exception()
        self.color = color
        return


