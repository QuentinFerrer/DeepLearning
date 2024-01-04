"""
Function which allows you to :
- open the images from the OnePiece dataset
- transforms them into an array with the correct label 
- separates them into train, develop, test.


"""
import os, sys
import cv2
import numpy as np

PATH_DATASET=("C:\\Users\\Ferrer Quentin\\Documents\\DeepLearning\\data\\DatasetOnePiece")
class DataOnePiece:

    def __init__(self) -> None:
        self.dict_data=self._load_name_classes()
        

    def load_picture(self):
        
        for cle in self.dict_data.keys():
            path_folder=os.path.join(PATH_DATASET, cle)
            images = []
            extensions_valides = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            
            for fichier in os.listdir(path_folder):
                if os.path.isfile(os.path.join(path_folder, fichier)) and \
                        any(fichier.lower().endswith(ext) for ext in extensions_valides):
                    chemin_image = os.path.join(path_folder, fichier)
                    image_array = cv2.imread(chemin_image)

                    if image_array is not None:
                        images.append(image_array)

            self.dict_data[cle] = images

    def transform_to_array(self):
        label=0
        X= np.empty((3, 3))
        for key,value_list in self.dict_data.items():
            print(value_list)
            val=np.stack(value_list)
            X=np.concatenate((X, val), axis=0)
        pass


    def _load_name_classes(self) -> dict[str,list]:

        name_file=os.path.join(PATH_DATASET,"classnames.txt")
        with open(name_file, 'r') as file:
            val = file.readlines()
        new_val = [row.strip() for row in val]
    
        dict_output={}
        for val in new_val:
            dict_output[val]=[]
        return dict_output
    
if __name__=="__main__":

    test=DataOnePiece()
    test.load_picture()
    test.transform_to_array()