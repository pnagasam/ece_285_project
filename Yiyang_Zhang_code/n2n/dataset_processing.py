"""
https://github.com/ZhenyuTan/Noise2Noise-Cryo-EM-image-denoising/blob/master/noise2noise_model_for_natural_img/data_set_builder.py
This class provides the dataset for training and testing the noise2noise model.
It implements dunder method for dataset class
"""


from __future__ import print_function, division
import os
import torch
from skimage import io
from skimage.transform import resize
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvF


class Training_Dataset(Dataset):

    def __init__(self,image_dir,noise_param,image_size,add_noise =True, crop = True):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.noise_param = noise_param
        self.image_size = image_size
        self.add_noise = add_noise
        self.crop_img = crop


    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self,idx):
        image_name = os.path.join(self.image_dir,self.image_list[idx])
        img = io.imread(image_name)
        img_cropped = self.__crop_img(img,image_name)
        source_temp = self.__add_noise(img_cropped,image_name)
        source = tvF.to_tensor(source_temp)
        target_temp = self.__add_noise(img_cropped,image_name)
        target = tvF.to_tensor(target_temp)           


        return source , target

    def __add_noise(self, img, fname):
        '''add Gaussain noise'''
        try:
            h,w,c = img.shape
            if c!=3:
                raise ValueError
            std = np.random.uniform(0,self.noise_param)
            noise = np.random.normal(0,std,(h,w,c))
        except:
            os.remove(fname)

        noise_img_temp = img + noise
        noise_img = np.clip(noise_img_temp, 0, 255).astype(np.uint8)
        return noise_img
        
    def __crop_img(self,img,fname):
        '''crop the img '''
        try:
            h, w, c = img.shape
            if c!=3:
                raise ValueError
        except:
            os.remove(fname)
        
        new_h, new_w = self.image_size,self.image_size
        if min(h,w) <  self.image_size:
            img = resize(img,(self.image_size,self.image_size),preserve_range=True)
        try:
            h_r,w_r,c = img.shape
        except:
            h_r,w_r = img.shape

        top = np.random.randint(0,h_r-new_h+1)
        left = np.random.randint(0,w_r-new_w+1)
        cropped_img = img[top:top+new_h,left:left+new_w]

        return cropped_img

class Testing_Dataset(Dataset):

    def __init__(self,image_dir,image_size):
        self.test_dir = image_dir
        self.test_list = os.listdir(image_dir)
        self.test_image_size = image_size


    def __len__(self):
        return len(os.listdir(self.test_dir))

    def __getitem__(self,idx):
        image_name = os.path.join(self.test_dir,self.test_list[idx])
        img = io.imread(image_name)
        input_temp = self.__crop_img(img)
        # input_exdim = np.expand_dims(input_temp, axis=-1)
        img_input = tvF.to_tensor(input_temp)
  
        return img_input

    def __crop_img(self,img):
        '''crop the img '''
        h, w ,_= img.shape
        new_h, new_w = self.test_image_size,self.test_image_size
        top = np.random.randint(0,h-new_h+1)
        left = np.random.randint(0,w-new_w+1)
        cropped_img = img[top:top+new_h,left:left+new_w]

        return cropped_img


    




        