from integrator import Integrator


class MISIntegrator(Integrator):
    def ell(self, scene, ray):
        if (scene.intersect(ray)):
            # intersection point where object was hit
            intersPoint = ray.o + ray.d*ray.t





            return ray.firstHitShape


        return [0,0,0] # no intersection so we stare into the deep void
