from backgroundImageProcessor import BIP
import random
import math
import yaml
import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import random
from ImageTransformer import ImageTransformer
from ReverseDefisheye import ReverseDefisheye

MIN_BBOX_SIZE = 10  # Example minimum size for bounding boxes
class MakeImages:

    def __init__(self):
        self.crops = self.load_images_to_dict('Crops')
        self.belts = self.load_images_to_dict('undistortedBeltImages')
        self.bip = BIP()
        self.dataset_directory= ''
        self.data_yaml = ''



    def place_bbox_in_rotated_roi(self,roi_list, bbox):
        bbox_width, bbox_height = bbox

        # Precompute valid placement ranges for each ROI
        precomputed_ranges = []
        for roi in roi_list:
            center_x, center_y, roi_width, roi_height, angle = roi
            angle_rad = math.radians(angle)

            # Effective bounds considering the bounding box rotation
            rotated_width = abs(bbox_width * math.cos(angle_rad)) + abs(bbox_height * math.sin(angle_rad))
            rotated_height = abs(bbox_width * math.sin(angle_rad)) + abs(bbox_height * math.cos(angle_rad))

            # Ensure bounding box fits within the ROI
            if rotated_width > roi_width or rotated_height > roi_height:
                continue  # Skip this ROI since the bounding box won't fit

            # Define valid placement ranges in the ROI's local coordinate system
            x_min = -roi_width / 2 + rotated_width / 2
            x_max = roi_width / 2 - rotated_width / 2
            y_min = -roi_height / 2 + rotated_height / 2
            y_max = roi_height / 2 - rotated_height / 2

            precomputed_ranges.append((center_x, center_y, x_min, x_max, y_min, y_max, angle_rad))

        # Try placing the bounding box within the precomputed valid ranges
        for attempt in range(100):
            for center_x, center_y, x_min, x_max, y_min, y_max, angle_rad in precomputed_ranges:
                # Generate random offsets within valid ranges
                offset_x = random.uniform(x_min, x_max)
                offset_y = random.uniform(y_min, y_max)

                # Compute the top-left corner of the bounding box in the ROI's local coordinate system
                x_local = offset_x - bbox_width / 2
                y_local = offset_y - bbox_height / 2

                # Convert local coordinates to global coordinates
                x_global = center_x + x_local * math.cos(angle_rad) - y_local * math.sin(angle_rad)
                y_global = center_y + x_local * math.sin(angle_rad) + y_local * math.cos(angle_rad)

                # Compute bottom-right corner
                x2_global = x_global + bbox_width * math.cos(angle_rad) - bbox_height * math.sin(angle_rad)
                y2_global = y_global + bbox_width * math.sin(angle_rad) + bbox_height * math.cos(angle_rad)

                # Ensure x1 < x2 and y1 < y2 by swapping if necessary
                x1, x2 = sorted([int(x_global), int(x2_global)])
                y1, y2 = sorted([int(y_global), int(y2_global)])

                # Check that the bounding box fits entirely within the ROI in global coordinates
                if (
                    x2 - x1 >= MIN_BBOX_SIZE and y2 - y1 >= MIN_BBOX_SIZE and
                    center_x - roi_width / 2 <= x1 <= center_x + roi_width / 2 and
                    center_y - roi_height / 2 <= y1 <= center_y + roi_height / 2 and
                    center_x - roi_width / 2 <= x2 <= center_x + roi_width / 2 and
                    center_y - roi_height / 2 <= y2 <= center_y + roi_height / 2
                ):
                    return [x1, y1, x2, y2], False

        # Return None if no valid placement is found
        return None,True






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
                if os.path.exists(f'{directory}/data.yaml'):
                    # d = input("Selected dataset directory is not empty. and contains data.yaml, New data will be added to this dataset. Do you want to continue? (y/n): ")
                    d= 'y'
                    if d.lower() == 'y':
                        with open(f"{directory}/data.yaml", "r") as file:
                            self.data_yaml = yaml.safe_load(file)  # Use safe_load to avoid arbitrary code execution
                            self.dataset_directory = directory 
                    else:
                        newDirName = input("Enter new directory name : ")
                        self.createDatasetDirectory(newDirName)
                    return
                else:
                        newDirName = input("the dataset directory is present but does not contain data.yaml hence is not usable. Enter new directory name : ")
                        self.createDatasetDirectory(newDirName)

        else:
            print(f"Creating Dataset Directory {directory}")
            self
            os.makedirs(f"{directory}/images/train", exist_ok=True)
            os.makedirs(f"{directory}/labels/train", exist_ok=True)
            os.makedirs(f"{directory}/images/val", exist_ok=True)
            os.makedirs(f"{directory}/labels/val", exist_ok=True)
            self.dataset_directory = directory 
            self.data_yaml = {
                'path':directory,
                'train': 'images/train',
                'val': 'images/val',
                # Number of classes
                'nc': 0,
                'names':[]
            }
            with open(f"{directory}/data.yaml", "w") as file:
                yaml.dump(self.data_yaml, file, default_flow_style=False)  

    def resize_by_factor(self,image,factor):
        image = cv2.resize(image,(0,0),fx=factor,fy=factor)
        return image

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
    
    def get_or_add_index(self,array, element):
        """
        Get the index of an element in an array. 
        If the element is not present, add it to the array and return its index.
        
        Args:
            array (list): The array to search or modify.
            element (Any): The element to find or add.
        
        Returns:
            int: The index of the element in the array.
        """
        if element in array:
            return array.index(element)
        else:
            array.append(element)
            return len(array) - 1
        


    def makeImage(self,classes_to_skip=[]):
        data = np.load('defisheye_maps.npz')

        # Access individual matrices
        xs = data['xs']
        ys = data['ys']

        transformer = ImageTransformer()
        refisher = ReverseDefisheye()

        for clsName,img in self.crops.items():
            if clsName in classes_to_skip:
                continue

            cls = clsName.split('_')[0]
            cls_id = self.get_or_add_index(self.data_yaml['names'],cls)
            self.data_yaml['nc']=len(self.data_yaml['names'])
            for ip,belt_ref in self.belts.items():
                for folder in ['train','val']:
                    num = 3 if folder == 'train' else 1
                    for i in range(0,num):
                        belt=belt_ref.copy()
                        img_copy = img.copy()
                        resize_factor=self.bip.getScaleFactor(ip)
                        tryPlacement = True
                        numTries = 0
                        while tryPlacement and numTries<11:
                            numTries += 1
                            transformed_img = transformer.random_transformation(img)
                            h,w = transformed_img.shape[:2]
                            w_res = w*resize_factor
                            h_res = h*resize_factor
                            # hb,wb = belt.shape[:2]
                            rois = self.bip.getRois(ip)
                            print(ip,rois)
                            bbox,tryPlacement = self.place_bbox_in_rotated_roi(rois,[w_res,h_res])
                            # bbox = [(wb-w)//2,(hb-h)//2,(wb+w)//2,(hb+h)//2]
                            if bbox:
                                combined = self.mergeImages(belt,transformed_img,bbox)
                                defishParams=self.bip.getdefishParams(ip)
                                padding_info = self.bip.getPadding_info(ip)
                                if defishParams:
                                    fov,pfov =defishParams[0],defishParams[1]
                                else:
                                    fov = 145
                                    pfov = 128
                                refished = refisher.distort(combined,fov=fov,pfov=pfov)
                                h,w = refished.shape[:2]
                                x1,y1,x2,y2 = bbox
                                x_tl = int(ys[x1,y1])
                                x_bl = int(ys[x1,y2])
                                x_tr = int(ys[x2,y1])
                                x_br = int(ys[x2,y2])
                                y_tl = int(xs[x1,y1])
                                y_bl = int(xs[x1,y2])
                                y_tr = int(xs[x2,y1])
                                y_br = int(xs[x2,y2])

                                x1_t,y1_t = (min(x_tl,x_bl),min(y_tl,y_tr))
                                x2_t,y2_t = (max(x_tr,x_br),max(y_bl,y_br))

                                # x1_t,y1_t = refisher.get_distorted_coordinates(y1,x1,fov=fov,pfov=pfov,width=w,height=h)
                                # x2_t,y2_t = refisher.get_distorted_coordinates(y2,x2,fov=fov,pfov=pfov,width=w,height=h)
                                x1_t -= padding_info['left']
                                x2_t -= padding_info['left']
                                y1_t -= padding_info['top']
                                y2_t -= padding_info['top']

                                depadded = self.bip.remove_padding(refished,ip,padding_info)
                                w_t = w - padding_info['left'] - padding_info['right']
                                h_t = h - padding_info['top'] - padding_info['bottom']
                                x_center = (x1_t + x2_t) / 2
                                y_center = (y1_t + y2_t) / 2
                                width = x2_t - x1_t
                                height = y2_t - y1_t

                                cv2.imwrite(f'{self.dataset_directory}/images/{folder}/{ip}_{clsName}_{i}.jpg',depadded)
                                with open(f"{self.dataset_directory}/data.yaml", "w") as file:
                                    yaml.dump(self.data_yaml, file, default_flow_style=False)  
                                with open(f'{self.dataset_directory}/labels/{folder}/{ip}_{clsName}_{i}.txt','w') as f:
                                    f.writelines(f'{cls_id} {x_center/w_t} {y_center/h_t} {width/w_t} {height/h_t}')
                                # cv2.rectangle(depadded, (x1_t,y1_t),(x2_t,y2_t), (255, 0, 0) , thickness=4)
                                # self.bip.show_images_sidebyside(combined,refished,depadded,transformed_img)
                    
                    # return




if __name__=='__main__':
    imagemaker = MakeImages()
    imagemaker.createDatasetDirectory('dataset7')
    
    classes_to_skip=['bag1','bag2']
    imagemaker.makeImage(classes_to_skip)

