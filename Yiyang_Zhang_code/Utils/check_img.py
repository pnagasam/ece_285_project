import os
from PIL import Image
import numpy as np


def delete_non_rgb_images(folder_path):
    # Loop through all files in the folder
    counter = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    pimg = np.array(img)
                    a,b,c = pimg.shape
                    if c != 3:
                        raise ValueError
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                os.remove(file_path)
                counter += 1
    print(f'deleted {counter} images')


if __name__ == '__main__':
    folder_path = '../n2n/data/train'
    delete_non_rgb_images(folder_path)