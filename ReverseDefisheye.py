import cv2
from numpy import arange, sqrt, arctan, sin, tan, meshgrid, pi, pad
from numpy import ndarray, hypot
import numpy as np

class ReverseDefisheye:
    """
    ReverseDefisheye

    This class takes an undistorted image along with the parameters that were
    used to undistort it and applies fisheye distortion back to the image.

    Parameters:
    fov: fisheye field of view (aperture) in degrees
    pfov: perspective field of view (aperture) in degrees
    xcenter: x center of fisheye area
    ycenter: y center of fisheye area
    radius: radius of fisheye area
    pad: Expand image in width and height
    angle: image rotation in degrees clockwise
    dtype: linear, equalarea, orthographic, stereographic
    format: circular, fullframe
    """

    def distort(self, undistorted_image, **kwargs):
        vkwargs = {"fov": 140,
                   "pfov": 120,
                   "xcenter": None,
                   "ycenter": None,
                   "radius": None,
                   "pad": 0,
                   "angle": 0,
                   "dtype": "equalarea",
                   "format": "fullframe"
                   }
        self._start_att(vkwargs, kwargs)

        if type(undistorted_image) == str:
            _image = cv2.imread(undistorted_image)
        elif type(undistorted_image) == ndarray:
            _image = undistorted_image
        else:
            raise Exception("Image format not recognized")

        self._width = _image.shape[1]
        self._height = _image.shape[0]

        if self._xcenter is None:
            self._xcenter = (self._width - 1) // 2

        if self._ycenter is None:
            self._ycenter = (self._height - 1) // 2

        distorted_image = self._apply_distortion(_image)
        return distorted_image

    def _start_att(self, vkwargs, kwargs):
        """
        Starting attributes
        """
        pin = []

        for key, value in kwargs.items():
            if key not in vkwargs:
                raise NameError("Invalid key {}".format(key))
            else:
                pin.append(key)
                setattr(self, "_{}".format(key), value)

        pin = set(pin)
        rkeys = set(vkwargs.keys()) - pin
        for key in rkeys:
            setattr(self, "_{}".format(key), vkwargs[key])

    def _apply_distortion(self, image):
        if self._format == "circular":
            dim = min(self._width, self._height)
        elif self._format == "fullframe":
            dim = sqrt(self._width ** 2.0 + self._height ** 2.0)

        if self._radius is not None:
            dim = 2 * self._radius

        ofoc = dim / (2 * tan(self._pfov * pi / 360))

        i = arange(self._width)
        j = arange(self._height)
        i, j = meshgrid(i, j)

        xd, yd = self._reverse_map(i, j, ofoc, dim)

        distorted_image = cv2.remap(image, xd, yd, cv2.INTER_LINEAR)
        return distorted_image

    def _reverse_map(self, i, j, ofoc, dim):
        xd = i - self._xcenter
        yd = j - self._ycenter

        rd = hypot(xd, yd)

        if self._dtype == "linear":
            ifoc = dim * 180 / (self._fov * pi)
            phiang = rd / ifoc

        elif self._dtype == "equalarea":
            ifoc = dim / (2.0 * sin(self._fov * pi / 720))
            phiang = 2 * np.arcsin(rd / ifoc)

        elif self._dtype == "orthographic":
            ifoc = dim / (2.0 * sin(self._fov * pi / 360))
            phiang = np.arcsin(rd / ifoc)

        elif self._dtype == "stereographic":
            ifoc = dim / (2.0 * tan(self._fov * pi / 720))
            phiang = 2 * np.arctan(rd / ifoc)

        rr = ofoc * np.tan(phiang)

        xs = xd.astype(np.float32).copy()
        ys = yd.astype(np.float32).copy()

        rdmask = rd != 0
        xs[rdmask] = (rr[rdmask] / rd[rdmask]) * xd[rdmask] + self._xcenter
        ys[rdmask] = (rr[rdmask] / rd[rdmask]) * yd[rdmask] + self._ycenter

        xs[~rdmask] = 0
        ys[~rdmask] = 0
        return xs, ys
