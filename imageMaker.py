from backgroundImageProcessor import BIP
import random
import math

import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
MIN_BBOX_SIZE = 5 
class MakeImages:

    def __init__(self):
        self.crops = self.load_images_to_dict('Crops')
        self.belts = self.load_images_to_dict('undistortedBeltImages')
        self.bip = BIP()



    def place_bbox_in_roi(self, roi_list, bbox):
        bbox_width, bbox_height = bbox

        # Precompute valid placement ranges for each ROI
        precomputed_ranges = []
        for roi in roi_list:
            center_x, center_y, roi_width, roi_height, angle = roi
            angle_rad = math.radians(angle)

            # Define the effective bounds within the ROI that can accommodate the bounding box
            x_min = -roi_width / 2 + bbox_width / 2
            x_max = roi_width / 2 - bbox_width / 2
            y_min = -roi_height / 2 + bbox_height / 2
            y_max = roi_height / 2 - bbox_height / 2

            if x_min <= x_max and y_min <= y_max:
                precomputed_ranges.append((center_x, center_y, x_min, x_max, y_min, y_max, angle_rad))

        # Try placing the bounding box within the precomputed valid ranges
        for attempt in range(100):
            for center_x, center_y, x_min, x_max, y_min, y_max, angle_rad in precomputed_ranges:
                offset_x = random.uniform(x_min, x_max)
                offset_y = random.uniform(y_min, y_max)

                # Compute the top-left corner of the bounding box in the ROI's local coordinate system
                x_local = offset_x - bbox_width / 2
                y_local = offset_y - bbox_height / 2

                # Convert local coordinates to global coordinates
                x_global = center_x + x_local * math.cos(angle_rad) - y_local * math.sin(angle_rad)
                y_global = center_y + x_local * math.sin(angle_rad) + y_local * math.cos(angle_rad)

                # Compute bottom-right corner
                x2_global = x_global + bbox_width * math.cos(angle_rad)
                y2_global = y_global + bbox_height * math.sin(angle_rad)

                # Ensure x1 < x2 and y1 < y2 by swapping if necessary
                x1, x2 = sorted([int(x_global), int(x2_global)])
                y1, y2 = sorted([int(y_global), int(y2_global)])

                # Calculate bounding box coordinates and check that it fits within the ROI
                if (
                    x2 - x1 >= MIN_BBOX_SIZE and y2 - y1 >= MIN_BBOX_SIZE and
                    center_x - roi_width / 2 <= x1 <= center_x + roi_width / 2 and
                    center_y - roi_height / 2 <= y1 <= center_y + roi_height / 2 and
                    center_x - roi_width / 2 <= x2 <= center_x + roi_width / 2 and
                    center_y - roi_height / 2 <= y2 <= center_y + roi_height / 2
                ):
                    return [x1, y1, x2, y2]

        return None




    def load_images_to_dict(self,directory_path):
        """
        Load all images from the specified directory into a dictionary.
        The key is the image file name without the extension, and the value is the image as a NumPy array.

        Parameters:
            directory_path (str): Path to the directory containing images.

        Returns:
            dict: Dictionary with image names as keys and images as values.
        """
        image_dict = {}

        # List all files in the directory
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            
            # Check if the file is an image
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Load the image using OpenCV
                image = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
                
                if image is not None:
                    # Use the file name without extension as the key
                    image_name = os.path.splitext(file_name)[0]
                    image_dict[image_name] = image
        
        return image_dict

    def createDatasetDirectory(self, directory):
        if os.path.isdir(directory):
            if os.listdir(directory):  # Check if the directory is not empty
                # d = input("Selected dataset directory is not empty. Continuing will delete the previous dataset. Do you want to continue? (y/n): ")
                d= 'y'
                if d.lower() == 'y':
                    shutil.rmtree(directory)  # Delete the directory and its contents
                    self.createDatasetDirectory(directory)
                else:
                    newDirName = input("Enter new directory name : ")
                    self.createDatasetDirectory(newDirName)
                return
        else:
            print(f"Creating Dataset Directory {directory}")
            os.makedirs(f"{directory}/train/images", exist_ok=True)
            os.makedirs(f"{directory}/train/labels", exist_ok=True)
            os.makedirs(f"{directory}/val/images", exist_ok=True)
            os.makedirs(f"{directory}/val/labels", exist_ok=True)

    def mergeImages(self,background ,foreground ,bbox):
        x1,y1,x2,y2 = bbox 
        foreground_resized = cv2.resize(foreground, (x2 - x1, y2 - y1))
        b, g, r, alpha = cv2.split(foreground_resized)


        # Normalize the alpha mask to the range [0, 1]
        alpha = alpha / 255.0

        # Extract the region of interest (ROI) from the background where the foreground will be placed
        roi = background[y1:y2, x1:x2]

        # Blend the foreground and the ROI of the background
        blended_roi = cv2.convertScaleAbs(foreground_resized[:, :, :3] * alpha[:, :, None] +
                                        roi * (1 - alpha[:, :, None]))

        # Replace the ROI on the background with the blended ROI
        background[y1:y2, x1:x2] = blended_roi
        return background
    

    def makeImage(self):
        for clsName,img in self.crops.items():
            h,w = img.shape[:2]
            for ip,belt in self.belts.items():
                rois = self.bip.getRois(ip)
                print(ip,rois)
                bbox = self.place_bbox_in_roi(rois,[w,h])
                if bbox:
                    combined = self.mergeImages(belt,img,bbox)
                    
                    plt.imshow(combined)
                    plt.show()
                    return




if __name__=='__main__':
    imagemaker = MakeImages()
    imagemaker.createDatasetDirectory('dataset1')
    imagemaker.makeImage()

