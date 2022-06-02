import os
import cv2
from PIL import Image
from pathlib import Path


path = "Data_PNG"
cell_types = ["HSIL", "LSIL", "NSIL"]

for type in cell_types:
    list = os.listdir(f"{path}/{type}")
    print(f"{type} image count: {len(list)}")
    # out_dir = f"Data_Conv/{type}"

    # new_imgs = []
    # base_shape = (1130, 1130, 3)
    # for img in list(Path(path).joinpath(type).glob('*.png')):
    #     image = cv2.imread(str(img))
    #     if image.shape != base_shape:
    #         new_imgs.append((cv2.resize(image, (base_shape[:2])), img))
    # for im, path in new_imgs:
    #     cv2.imwrite(str(path), im)  
