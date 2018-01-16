
from Materials.Material import Material


class PhongMaterial(Material) :


    def __init__(self, color, Kdiffuse, Kspecular, Kshininess ) :
        Material.__init__(self, color)
        self.Kdiff = Kdiffuse
        self.Kspec = Kspecular
        self.Kshini = Kshininess
        return


    def shade(self):
        # todo implement phong
        """
                     // normal on intersection point
            Vec3d N = intersection.normal();

            //vector from intersection to light
            Vec3d L = light.position() - intersection.position();

        //light intensity decreases with the square of the distance to the illuminated object
            double lightStrength = light.spectralIntensity().length() / (L.length() * L.length());
            Vec3d lightColor = Vec3d(light.spectralIntensity());

            //we have to normalize all vectors going into the phong calculations
            L.normalize();


            //reflected ray cast from light
            Vec3d R = (2 * dot(L, N) * N) - L;

            //vector from intersection to viewer
            Vec3d omega0 = intersection.ray().origin() - intersection.position();

            //specular component for the color at the current point
            double specularLight = dot(omega0.normalize(), R.normalize());
            //diffuse component
            double diffuseLight = std::max(dot(N, L), double(0));

            //multiplier for the components to have a bit of control over their intensity
            double Kdiffuse = 0.2;
            double Kspecular = 0.1;
            //coefficient to counter the shrinking of all specular highlights caused by high shininess exponents
            double specularNormalizationFactor = (mShininess + 2) / (2 * M_PI);

            //combining all light components
            double fr = diffuseLight * Kdiffuse * lightStrength +
                        (Kspecular * pow(specularLight, mShininess) * specularNormalizationFactor * lightStrength);

            //phong lightning really only calculates how much a fragment should be made darker or lighter
            //so to take the light color into account, we have to mix the light and object color fist and then highlight with phong
            //(1.0/255.0) factor to make the light colors go from 0-1; chaning the intensity in raytracer_task3.cpp will break color
            return Vec4d(PhongMaterial::mixColor(this->color(), lightColor * (1.0 / 255.0)) * fr, 1.0);
        """
        return

