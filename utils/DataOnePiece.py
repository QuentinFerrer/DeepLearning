"""
Function which allows you to :
- open the images from the OnePiece dataset
- transforms them into an array with the correct label 
- separates them into train, develop, test.


"""
import os
import random
import sys
from typing import Dict

import cv2
import numpy as np
import tqdm


class DataOnePiece:
    def __init__(self):
        pass

    def load_split_images(self, path_data: str, path_images_resized: str):
        dict_data = self._get_keys(path_data)
        dict_data = self._get_images(dict_data, path_images_resized)
        dict_data = self._shuffle_listes_dictionnaire(dict_data, 98)
        dict_train, dict_dvlp, dict_test = self._split_train_dev_test(
            dict_data, 500, 100
        )
        x_train, y_train = self._dict_to_array(dict_train)
        x_dev, y_dev = self._dict_to_array(dict_dvlp)
        x_test, y_test = self._dict_to_array(dict_test)
        return (x_train, y_train, x_dev, y_dev, x_test, y_test)

    def load_resize_save_images(
        self, path_data: str, path_images_init: str, path_output: str
    ):
        dict_data = self._get_keys(path_data)
        dict_data = self._get_images(dict_data, path_images_init)
        dict_data = self._resize_images(dict_data, shape=(500, 500))
        self._save_images(dict_data, path_output)

    def _dict_to_array(self, dict_data):
        X = []
        Y = []

        for label, liste_images in tqdm.tqdm(dict_data.items()):
            for image in liste_images:
                X.append(image)
                Y.append(self._correspondance_label(label))

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def _correspondance_label(self, label_txt: str) -> int:
        dict_labels = {
            "Ace": 0,
            "Akainu": 1,
            "Brook": 2,
            "Chopper": 3,
            "Crocodile": 4,
            "Franky": 5,
            "Jinbei": 6,
            "Kurohige": 7,
            "Law": 8,
            "Luffy": 9,
            "Mihawk": 10,
            "Nami": 11,
            "Rayleigh": 12,
            "Robin": 13,
            "Sanji": 14,
            "Shanks": 15,
            "Usopp": 16,
            "Zoro": 17,
        }
        return dict_labels[label_txt]

    def _split_train_dev_test(self, dict_data, len_train, len_develop):
        dict_train = {}
        dict_dvlp = {}
        dict_test = {}
        for cle in dict_data:
            dict_train[cle] = dict_data[cle][:len_train]
            dict_dvlp[cle] = dict_data[cle][len_train:len_develop]
            dict_test[cle] = dict_data[cle][len_develop:]
        return dict_train, dict_dvlp, dict_test

    def _shuffle_listes_dictionnaire(self, dict_data, seed):
        random.seed(seed)
        for cle, valeur in dict_data.items():
            random.shuffle(valeur)
        return dict_data

    def _save_images(self, dict_data, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for cle, images in tqdm.tqdm(dict_data.items()):
            folder_path = os.path.join(output_folder, cle)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for i, image_array in enumerate(images):
                nom_image = f"image_{i}.jpg"  # Nom de l'image, peut être ajusté selon vos besoins
                chemin_sortie = os.path.join(folder_path, nom_image)
                cv2.imwrite(chemin_sortie, image_array)

    def _resize_images(self,dict_data, shape=(500, 500)) -> Dict[str, list]:
        dict_data_resized = {}

        for cle, liste_arrays in dict_data.items():
            dict_data_resized[cle] = []
            for array in liste_arrays:
                image = cv2.cvtColor(
                    array, cv2.COLOR_RGB2BGR
                )  # Convertir de RGB à BGR pour OpenCV
                hauteur, largeur, _ = image.shape
                facteur_redim = min(shape[0] / hauteur, shape[1] / largeur)
                nouvelle_taille = (
                    int(largeur * facteur_redim),
                    int(hauteur * facteur_redim),
                )
                image_redimensionnee = cv2.resize(
                    image, nouvelle_taille, interpolation=cv2.INTER_AREA
                )
                image_fixe = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                debut_hauteur = (shape[0] - nouvelle_taille[1]) // 2
                debut_largeur = (shape[1] - nouvelle_taille[0]) // 2
                image_fixe[
                    debut_hauteur : debut_hauteur + nouvelle_taille[1],
                    debut_largeur : debut_largeur + nouvelle_taille[0],
                ] = image_redimensionnee
                dict_data_resized[cle].append(image_fixe)

        return dict_data_resized

    def _get_images(
        self, dict_data: Dict[str, list], path_images_init: str
    ) -> Dict[str, list]:
        for cle in tqdm.tqdm(dict_data.keys()):
            path_folder = os.path.join(path_images_init, cle)
            images = []
            extensions_valides = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

            for fichier in os.listdir(path_folder):
                if os.path.isfile(os.path.join(path_folder, fichier)) and any(
                    fichier.lower().endswith(ext) for ext in extensions_valides
                ):
                    chemin_image = os.path.join(path_folder, fichier)
                    image_array = cv2.imread(chemin_image)

                    if image_array is not None:
                        images.append(image_array)

            dict_data[cle] = images
        return dict_data

    def _get_keys(self, path_data) -> dict:
        name_file = os.path.join(path_data, "classnames.txt")
        with open(name_file, "r") as file:
            val = file.readlines()
        new_val = [row.strip() for row in val]
        dict_data = {}
        for val in new_val:
            dict_data[val] = []
        return dict_data


if __name__ == "__main__":
    path_data = (
        "C:\\Users\\Ferrer Quentin\\Documents\\DeepLearning\\data\\DatasetOnePiece"
    )
    path_images_init = "C:\\Users\\Ferrer Quentin\\Documents\\DeepLearning\\data\\DatasetOnePiece\\init"
    path_images_resized = "C:\\Users\\Ferrer Quentin\\Documents\\DeepLearning\\data\\DatasetOnePiece\\data_resized"

    Data = DataOnePiece()
    Data.load_resize_save_images(path_data, path_images_init, path_images_resized)
    res = Data.load_split_images(path_data, path_images_resized)
