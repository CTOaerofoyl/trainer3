from defisheye import Defisheye
from ReverseDefisheye import ReverseDefisheye

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(*images, titles=None):
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


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: ({x}, {y})")
        return (x,y)


def make_square(image):
    height, width = image.shape[:2]
    top, bottom, left, right = 0, 0, 0, 0
    
    # Calculate padding size for each dimension
    if height > width:
        padding = (height - width) // 2
        left, right = padding, height - width - padding
        padded_image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif width > height:
        padding = (width - height) // 2
        top, bottom = padding, width - height - padding
        padded_image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        # The image is already square
        padded_image = image

    padding_info = {"top": top, "bottom": bottom, "left": left, "right": right}
    return padded_image, padding_info

# Load your image

def get_original_image(padded_image, padding_info):
    top = padding_info["top"]
    bottom = padding_info["bottom"]
    left = padding_info["left"]
    right = padding_info["right"]
    
    original_image = padded_image[top:padded_image.shape[0] - bottom, left:padded_image.shape[1] - right]
    return original_image






cap = cv2.VideoCapture('rtsp://admin:PAG00319@192.168.1.221:554/live')
ret,frame = cap.read()
deFishApp = Defisheye()
reFishApp= ReverseDefisheye()
fov = 145
pfov = 128
# Make the image square
square_image,padding_info = make_square(frame)

img = deFishApp.convert(square_image,fov= fov,pfov=pfov)

img2 = reFishApp.distort(img,fov= fov,pfov=pfov)

print(square_image.shape[:2],img.shape[:2],img2.shape[:2])

show(square_image,img,img2,titles=['original','Undistorted','Redistorted'])

im3 = get_original_image(img2, padding_info)

show(frame,im3)