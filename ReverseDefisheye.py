import cv2
from numpy import arange, sqrt, arctan, sin, tan, meshgrid, pi, pad
from numpy import ndarray, hypot
import numpy as np
import matplotlib.pyplot as plt
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
    def show_images_sidebyside(self,*images, titles=None):
        """
        Displays any number of images side by side.
        
        Parameters:
            images: Any number of image arrays to display.
            titles: A list of titles corresponding to the images (optional).
        """
        n = len(images)
        if n == 0:
            raise ValueError("No images provided to display.")
        
        # Create a figure with n columns
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        
        # If only one image, axes is not iterable, so we wrap it in a list
        if n == 1:
            axes = [axes]
        
        # Loop through the images and display each
        for i, (image, ax) in enumerate(zip(images, axes)):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:  # Check if it's a color image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ax.imshow(image)
            ax.axis('off')  # Turn off axis
            if titles and i < len(titles):
                ax.set_title(titles[i])  # Set title if provided
        
        plt.tight_layout()
        plt.show()

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

        xs[~rdmask] = self._xcenter
        ys[~rdmask] = self._ycenter
        return xs, ys

    def get_distorted_coordinates(self, x, y, **kwargs):
        """
        Get distorted coordinates for a given (x, y).

        Parameters:
        x: x-coordinate in the original (undistorted) image
        y: y-coordinate in the original (undistorted) image
        width: Width of the image
        height: Height of the image

        Returns:
        (x_distorted, y_distorted): Coordinates in the distorted image
        """
        width = kwargs.pop('width', 1000)
        height = kwargs.pop('height', 1000)

        self._width = width
        self._height = height

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

        if self._xcenter is None:
            self._xcenter = (self._width - 1) // 2

        if self._ycenter is None:
            self._ycenter = (self._height - 1) // 2

        if self._format == "circular":
            dim = min(self._width, self._height)
        elif self._format == "fullframe":
            dim = sqrt(self._width ** 2.0 + self._height ** 2.0)

        if self._radius is not None:
            dim = 2 * self._radius

        ofoc = dim / (2 * tan(self._pfov * pi / 360))
        i = arange(width)
        j = arange(height)
        i, j = meshgrid(i, j)
        # Reuse the same reverse mapping logic for accuracy
        xd, yd = self._reverse_map(i,j, ofoc, dim)
        tolerance = 1
        matches = np.where(np.isclose(xd, x, atol=tolerance) & np.isclose(yd, y, atol=tolerance))
        x_distorted, y_distorted=matches[0][0], matches[1][0]


        return x_distorted, y_distorted

# def test_reverse_defisheye():
#     """
#     Test function for the ReverseDefisheye class.
#     """
#     # Create an instance of ReverseDefisheye
#     refish = ReverseDefisheye()

#     l = 2000
#     # Load a test image
#     undistorted_image = np.zeros((l, l, 3), dtype=np.uint8)
#     test_points = [(500, 500), (400, 400), (300, 300), (100, 100)]
#     undistorted_image[1:4,1:4,:]=255
#     distorted_coord = refish.get_distorted_coordinates(1,1, fov=140, pfov=120, width=l, height=l)
#     for point in test_points:
#         cv2.circle(undistorted_image, point, 10, (255, 0, 0), -1)

#     # # Distort the image
#     distorted_image = refish.distort(undistorted_image, fov=140, pfov=120)

#     # # Test multiple coordinates
#     for original_coord in test_points:
#         distorted_coord = refish.get_distorted_coordinates(*original_coord, fov=140, pfov=120, width=l, height=l)

#         # Visualize results
#         cv2.circle(distorted_image, (int(distorted_coord[0]), int(distorted_coord[1])), 5, (0, 255, 0), -1)
#         print(f"Original: {original_coord}, Distorted: {distorted_coord}")

#     refish.show_images_sidebyside(undistorted_image,distorted_image)

#     # cv2.imshow("Original and Distorted", np.hstack([undistorted_image, distorted_image]))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

# # Uncomment the following line to run the test
# # test_reverse_defisheye()


# # Uncomment the following line to run the test
# test_reverse_defisheye()
