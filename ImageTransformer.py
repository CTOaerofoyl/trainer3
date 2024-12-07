import random
import numpy as np
from PIL import Image
import cv2

class ImageTransformer:
    def random_transformation(self, image):
        # Convert image to RGBA
        if image.shape[2] == 3:  # If no alpha channel, add one
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Choose random transformations to apply
        transformations = ['rotate', 'flip', 'skew']
        chosen_transformations = random.sample(transformations, random.randint(1, 3))
        
        for transform in chosen_transformations:
            if transform == 'rotate':
                angle = random.uniform(-180, 180)
                image = self.rotate_image(image, angle)
            elif transform == 'flip':
                flip_type = random.choice(['horizontal', 'vertical'])
                image = self.flip_image(image, flip_type)
            elif transform == 'skew':
                skew_angle = random.uniform(-10, 10)
                image = self.skew_image(image, skew_angle)
        
        # Crop the image to remove excess empty space
        image = self.crop_to_content(image)
        
        return image

    @staticmethod
    def rotate_image(image, angle):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        
        # Calculate the new bounding dimensions of the rotated image
        radians = np.radians(angle)
        new_width = int(abs(np.sin(radians) * height) + abs(np.cos(radians) * width))
        new_height = int(abs(np.sin(radians) * width) + abs(np.cos(radians) * height))
        
        # Get the rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Adjust the rotation matrix to take into account the translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform the rotation with the new bounding dimensions
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
        return rotated_image

    @staticmethod
    def flip_image(image, flip_type):
        if flip_type == 'horizontal':
            flipped_image = cv2.flip(image, 1)
        else:
            flipped_image = cv2.flip(image, 0)
        
        return flipped_image

    @staticmethod
    def skew_image(image, skew_angle):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        
        # Define points for skewing
        src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
        skew_offset = width * np.tan(np.radians(skew_angle))
        dst_points = np.float32([[0, 0], [width - 1 + skew_offset, 0], [0, height - 1]])
        
        # Calculate new dimensions to accommodate skew
        new_width = int(width + abs(skew_offset))
        
        # Get the transformation matrix and apply the warp affine transformation
        transformation_matrix = cv2.getAffineTransform(src_points, dst_points)
        skewed_image = cv2.warpAffine(image, transformation_matrix, (new_width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
        return skewed_image

    @staticmethod
    def crop_to_content(image, debug=False):
        """
        Crops the given image to remove transparent borders caused by rotation.

        Parameters:
            image (numpy.ndarray): Input image (BGRA).
            debug (bool): Whether to display debugging visuals (default: False).

        Returns:
            numpy.ndarray: Cropped image.
        """
        # Extract the alpha channel to find non-transparent regions
        alpha_channel = image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

        # Find contours based on the non-transparent regions
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Calculate the bounding rectangle of the non-transparent area
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            cropped_image = image[y:y+h, x:x+w]
        else:
            # If no contours are found, return the original image
            cropped_image = image

        if debug:
            # Create a copy of the image to draw contours for debugging
            debug_image = image.copy()
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGRA2BGR)  # Remove alpha for visualization
            cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
            cv2.imshow('Debug - Contours', debug_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return cropped_image

# # Example usage
# transformer = ImageTransformer()
# input_image = cv2.imread('input_image.png', cv2.IMREAD_UNCHANGED)
# transformed_image = transformer.random_transformation(input_image)
# cv2.imwrite('output_image.png', transformed_image)
