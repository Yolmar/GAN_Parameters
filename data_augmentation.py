from distutils import extension
import os
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from numpy import save
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers

path = "Data_PNG"
out_path = "Data_Augmented_90k"
cell_types = ["HSIL", "LSIL", "NSIL"]

target_amount = 30000

print(f"Starting data augmentation...")
for type in cell_types:
    print(f"Currently working on {type} type...")

    l = os.listdir(f"{path}/{type}")
    print(f"{type} image count: {len(l)}")
    
    multiplier = int(target_amount / len(l))
    print(f"Target amount: {multiplier * len(l)}, multiplier: {multiplier}")

    for img in tqdm(list(Path(path).joinpath(type).glob('*.png'))):
        path_name, ext = os.path.splitext(img)
        file_name = path_name.split('/')
        img_name = file_name[-1]
        # print(f"Now augmenting -> {img_name}")

        image = cv2.imread(str(img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.expand_dims(image, 0)

        data_augmentation = tf.keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='nearest')])
        
        for i in range(multiplier):
            images = data_augmentation(image)
            image_aug = cv2.cvtColor(images[0].numpy(), cv2.COLOR_BGR2RGB)                        
            save_path = f"{out_path}/{type}/{img_name}_aug{i}.png"
            cv2.imwrite(str(save_path), image_aug)
