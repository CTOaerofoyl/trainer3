import random
import numpy as np
import cv2

class ImageTransformer:
    def random_transformation(self, image):
        # Convert image to RGBA
        if image.shape[2] == 3:  # If no alpha channel, add one
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Choose random transformations to apply
        transformations = ['rotate', 'flip', 'skew', 'blur', 'brightness_contrast']

        chosen_transformations = random.sample(transformations, random.randint(1, len(transformations)))

        for transform in chosen_transformations:
            if transform == 'rotate':
                angle = random.uniform(-45, 45)
                image = self.rotate_image(image, angle)
            elif transform == 'flip':
                flip_type = random.choice(['horizontal', 'vertical'])
                image = self.flip_image(image, flip_type)
            elif transform == 'skew':
                skew_angle = random.uniform(-10, 10)
                image = self.skew_image(image, skew_angle)
            elif transform == 'blur':
                image = self.random_blur(image)
            elif transform == 'brightness_contrast':
                image = self.adjust_brightness_contrast(image)
            elif transform == 'hue':
                image = self.adjust_hue(image)

        # Crop the image to remove excess empty space
        image = self.crop_to_content(image)

        return image

    @staticmethod
    def rotate_image(image, angle):
        height, width = image.shape[:2]
        radians = np.radians(angle)
        new_width = int(abs(np.sin(radians) * height) + abs(np.cos(radians) * width))
        new_height = int(abs(np.sin(radians) * width) + abs(np.cos(radians) * height))
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return rotated_image

    @staticmethod
    def flip_image(image, flip_type):
        return cv2.flip(image, 1 if flip_type == 'horizontal' else 0)

    @staticmethod
    def skew_image(image, skew_angle):
        height, width = image.shape[:2]
        src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
        skew_offset = width * np.tan(np.radians(skew_angle))
        dst_points = np.float32([[0, 0], [width - 1 + skew_offset, 0], [0, height - 1]])
        new_width = int(width + abs(skew_offset))
        transformation_matrix = cv2.getAffineTransform(src_points, dst_points)
        skewed_image = cv2.warpAffine(image, transformation_matrix, (new_width, height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return skewed_image

    @staticmethod
    def random_blur(image):
        kernel_size = random.randint(5, 15)
        angle = random.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        radians = np.radians(angle)
        dx = int(np.cos(radians) * (kernel_size // 2))
        dy = int(np.sin(radians) * (kernel_size // 2))
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness=1)
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    @staticmethod
    def adjust_brightness_contrast(image):
        alpha = random.uniform(0.8, 1.2)  # Contrast factor
        beta = random.randint(-30, 30)  # Brightness adjustment

        # Preserve transparency
        alpha_channel = image[:, :, 3]
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        adjusted = cv2.convertScaleAbs(bgr_image, alpha=alpha, beta=beta)
        adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2BGRA)
        adjusted[:, :, 3] = alpha_channel

        return adjusted

    @staticmethod
    def adjust_hue(image):
        if image.shape[2] == 4:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            alpha_channel = image[:, :, 3]
        else:
            bgr_image = image
            alpha_channel = None

        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        hue_shift = random.randint(-15, 15)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + hue_shift) % 180
        adjusted = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        if alpha_channel is not None:
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2BGRA)
            adjusted[:, :, 3] = alpha_channel

        return adjusted

    @staticmethod
    def crop_to_content(image, debug=False):
        alpha_channel = image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            cropped_image = image[y:y+h, x:x+w]
        else:
            cropped_image = image
        return cropped_image

# Example usage
# transformer = ImageTransformer()
# input_image = cv2.imread('input_image.png', cv2.IMREAD_UNCHANGED)
# transformed_image = transformer.random_transformation(input_image)
# cv2.imwrite('output_image.png', transformed_image)
