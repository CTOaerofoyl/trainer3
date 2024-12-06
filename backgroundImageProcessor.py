from defisheye import Defisheye
from ReverseDefisheye import ReverseDefisheye
import cv2
import json
import os
import matplotlib.pyplot as plt


class BIP():
    def __init__(self,saveCamConfig=True):
        self.saveCamConfig = saveCamConfig
        self.configFileName = 'camConfig1.json'
        self.configJSON = self.read_or_create_json(self.configFileName)
        self.defisher = Defisheye()
        self.reFishEye = ReverseDefisheye()

    def read_or_create_json(self,file_path):
        """
        Reads a JSON file. If the file does not exist, creates it with an empty dictionary.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: The content of the JSON file as a dictionary.
        """
        if not self.saveCamConfig:
            return {}
        if not os.path.exists(file_path):
            # Create the file with an empty dictionary
            with open(file_path, 'w') as file:
                json.dump({}, file, indent=4)
            print(f"File '{file_path}' created with an empty dictionary.")
            return {}
        
        # Read and return the content of the file
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def remove_padding(self,padded_image,ip, padding_info=None):
        if padding_info is None :
            camInit = False
            if ip in self.configJSON:
                camProps = self.configJSON[ip]
                if 'padding_info' in camProps:
                    padding_info= camProps['padding_info']
                    camInit = True
            if not camInit:
                print('This cam is not initialized yet, use make square dunction first to generate the padding info')
                return
            
        top = padding_info["top"]
        bottom = padding_info["bottom"]
        left = padding_info["left"]
        right = padding_info["right"]
        
        original_image = padded_image[top:padded_image.shape[0] - bottom, left:padded_image.shape[1] - right]
        return original_image
    
    def make_square(self,image,ip):
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
        if ip in self.configJSON:
            self.configJSON[ip]['padding_info']= padding_info
        else:
            self.configJSON[ip] = {'padding_info':padding_info}
        if self.saveCamConfig:
            with open(self.configFileName,'w') as file:
                json.dump(self.configJSON, file, indent=4) 

        return padded_image, padding_info
    
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
            ax.set_facecolor('lightyellow')
            fig.patch.set_facecolor('lightblue')
            ax.imshow(image)
            # ax.axis('off')  # Turn off axis
            if titles and i < len(titles):
                ax.set_title(titles[i])  # Set title if provided
        
        plt.tight_layout()
        plt.show()
    def defish(self,image,ip,fov = None,pfov = None):
        if fov is None or pfov is None:
            camInit = False
            if ip in self.configJSON:
                camProps = self.configJSON[ip]
                if 'defishParams' in camProps:
                    fov,pfov = camProps['defishParams']
                    camInit = True
            if not camInit:
                print('This cam is not initialized yet, provide fov and pfov values')
                return
        if ip in self.configJSON:
            self.configJSON[ip]['defishParams']= [fov,pfov]
        else:
            self.configJSON[ip] = {'defishParams': [fov,pfov]}
        if self.saveCamConfig:
            with open(self.configFileName,'w') as file:
                json.dump(self.configJSON, file, indent=4) 
        return self.defisher.convert(image,fov=fov,pfov=pfov)
    
    
    def refish(self,image,ip,fov = None,pfov = None):
        if fov is None or pfov is None:
            camInit = False
            if ip in self.configJSON:
                camProps = self.configJSON[ip]
                if 'defishParams' in camProps:
                    fov,pfov = camProps['defishParams']
                    camInit = True
            if not camInit:
                print('This cam is not initialized yet, provide camConfig file or reinitailize using defish')
                return
        return self.reFishEye.distort(image,fov=fov,pfov=pfov)
    
    def saveROIs(self,ip,ROIs):
        if ip in self.configJSON:
            self.configJSON[ip]['ROIs']= ROIs
        else:
            self.configJSON[ip] = {'ROIs': ROIs}

        with open(self.configFileName,'w') as file:
            json.dump(self.configJSON, file, indent=4) 
        
    def getRois(self,ip):
        if ip in self.configJSON:
            if 'ROIs' in self.configJSON[ip]:
                return self.configJSON[ip]['ROIs']
        return []
    
    def getScaleFactor(self,ip):
        if ip in self.configJSON:
            if 'resize_factor' in self.configJSON[ip]:
                return self.configJSON[ip]['resize_factor']
        return []
    
    def getdefishParams(self,ip):
        if ip in self.configJSON:
            if 'defishParams' in self.configJSON[ip]:
                return self.configJSON[ip]['defishParams']
        return None
    
    def getPadding_info(self,ip):
        if ip in self.configJSON:
            if 'padding_info' in self.configJSON[ip]:
                return self.configJSON[ip]['padding_info']
        return None

    
    def resize_by_factor(self,image,factor):
        image = cv2.resize(image,(0,0),fx=factor,fy=factor)
        return image

    def calculateScaleFactor(self,ip,referenceImage,backgroundImage):
        factor = 1
        while True:
            try:
                tempImage = backgroundImage.copy()
                resized_imge = self.resize_by_factor(referenceImage,factor)
                rows, cols, channels = resized_imge.shape
                tempImage[0:rows, 0:cols] = resized_imge
                self.show_images_sidebyside(tempImage,titles=[f'scaled by {factor}'])
                factor = float(input("Enter resize factor. Enter q to finish"))
                
            except:
                print(factor)
                if ip in self.configJSON:
                    self.configJSON[ip]['resize_factor']= factor
                else:
                    self.configJSON[ip] = {'resize_factor':factor}

                with open(self.configFileName,'w') as file:
                    json.dump(self.configJSON, file, indent=4) 
                return factor

