import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


class RotatedRectangleDrawer:
    def __init__(self):
        """
        Initialize the drawer. It allows drawing rectangles with dynamic rotation.
        """
        self.rectangles = []  # Store rectangles as [(center_x, center_y, width, height, angle)]
        self.start_point = None
        self.end_point = None
        self.current_preview = None  # Preview rectangle
        self.current_rotation_angle = 0  # Current rotation angle
        self.ax = None
        self.is_rotating = False  # Flag for rotation mode

    def on_mouse_press(self, event):
        """
        Mouse press event handler to capture the starting point of the rectangle.
        """
        if event.inaxes:
            self.start_point = (event.xdata, event.ydata)
            self.is_rotating = False  # Start with normal drawing

    def on_mouse_release(self, event):
        """
        Mouse release event handler to finalize the rectangle.
        """
        if event.inaxes and self.start_point:
            self.end_point = (event.xdata, event.ydata)

            # Calculate rectangle parameters
            center_x = (self.start_point[0] + self.end_point[0]) / 2
            center_y = (self.start_point[1] + self.end_point[1]) / 2
            width = abs(self.start_point[0] - self.end_point[0])
            height = abs(self.start_point[1] - self.end_point[1])
            angle = self.current_rotation_angle  # Rotation angle (if applied)

            # Save the rectangle
            self.rectangles.append((center_x, center_y, width, height, angle))
            print(f"Rectangle added: Center=({center_x}, {center_y}), Width={width}, Height={height}, Angle={angle}")

            # Clear preview
            self.current_preview = None
            self.start_point = None
            self.end_point = None
            self.current_rotation_angle = 0  # Reset rotation angle
            plt.gcf().canvas.draw()

    def on_mouse_motion(self, event):
        """
        Mouse motion event handler to update the preview rectangle.
        """
        if event.inaxes and self.start_point:
            if not self.is_rotating:
                # Normal rectangle drawing
                self.end_point = (event.xdata, event.ydata)
                center_x = (self.start_point[0] + self.end_point[0]) / 2
                center_y = (self.start_point[1] + self.end_point[1]) / 2
                width = abs(self.start_point[0] - self.end_point[0])
                height = abs(self.start_point[1] - self.end_point[1])
                self.update_preview(center_x, center_y, width, height, 0)
            else:
                # Rotate the rectangle
                angle = np.degrees(np.arctan2(event.ydata - self.start_point[1], event.xdata - self.start_point[0]))
                self.current_rotation_angle = angle
                center_x = (self.start_point[0] + self.end_point[0]) / 2
                center_y = (self.start_point[1] + self.end_point[1]) / 2
                width = abs(self.start_point[0] - self.end_point[0])
                height = abs(self.start_point[1] - self.end_point[1])
                self.update_preview(center_x, center_y, width, height, angle)

    def on_key_press(self, event):
        """
        Key press event handler to enable rotation.
        """
        if event.key == 'shift' and self.start_point and self.end_point:
            self.is_rotating = True

    def on_key_release(self, event):
        """
        Key release event handler to disable rotation.
        """
        if event.key == 'shift':
            self.is_rotating = False

    def update_preview(self, center_x, center_y, width, height, angle):
        """
        Update the preview rectangle.
        """
        if self.current_preview:
            self.current_preview.remove()  # Remove existing preview

        angle_rad = np.radians(angle)
        dx = width / 2
        dy = height / 2

        # Compute corners of the rectangle
        corners = np.array([
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ])

        # Apply rotation
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 0] += center_x
        rotated_corners[:, 1] += center_y

        # Draw preview
        self.current_preview = Polygon(rotated_corners, closed=True, edgecolor='g', facecolor='none', linestyle='--')
        self.ax.add_patch(self.current_preview)
        plt.gcf().canvas.draw()

    def draw(self, image):
        """
        Display the image and enable rectangle drawing.

        Args:
            image (numpy.ndarray): The image to draw rectangles on.
        """
        self.image = image
        self.rectangles = []
        # Display the image
        fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.ax.set_title("Draw rectangles, hold 'Shift' to rotate")

        # Connect events
        fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        # Block execution until ROIs are finalized
        plt.show(block=True)
        return self.rectangles
