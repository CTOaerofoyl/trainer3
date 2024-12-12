
import os
import yaml
import cv2
from ImageTransformer import ImageTransformer

class dataseCreatorfromCrops:
    def __init__(self):
        self.data_yaml = None
        self.dataset_directory = None
        self.imageTransoformer = ImageTransformer()

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
    
    def createDataset(self):
        images = self.load_images_to_dict('Crops')
        folder = 'train'
        for clsName, img in images.items():
            for i in range(0,5):
                img = self.imageTransoformer.random_transformation(img)
                cls = clsName.split('_')[0]
                cls_id = self.get_or_add_index(self.data_yaml['names'],cls)
                self.data_yaml['nc']=len(self.data_yaml['names'])
                cv2.imwrite(f'{self.dataset_directory}/images/{folder}/{clsName}_{i}.png',img)
                h,w = img.shape[:2]

                with open(f"{self.dataset_directory}/data.yaml", "w") as file:
                    yaml.dump(self.data_yaml, file, default_flow_style=False)  
                with open(f'{self.dataset_directory}/labels/{folder}/{clsName}_{i}.txt','w') as f:
                    f.writelines(f'{cls_id} 0.5 0.5 1 1')



if __name__ == '__main__':
    cc = dataseCreatorfromCrops()
    cc.createDatasetDirectory('cropDataset1')
    cc.createDataset()

