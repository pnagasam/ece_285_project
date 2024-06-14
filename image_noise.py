#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


# In[4]:


np.random.seed(0)


# In[4]:


# import zipfile
# with zipfile.ZipFile("out_door_pics-20240530T235606Z-001.zip","r") as zip_ref:
#     zip_ref.extractall("ece285_photos")


# In[5]:


def choose_pics():
    for root, _, fnames in os.walk('ece285_photos'):
        for fname in fnames:
            if fname.endswith('.CR2'):
                fp = os.path.join(root, fname)
                im = plt.imread(fp)
                plt.imshow(im)
                print(fp)
                plt.show()
                
# choose_pics()


# In[3]:


pic_fnames = ['ece285_photos/out_door_pics/A56B9042.CR2', 'ece285_photos/out_door_pics/A56B9043.CR2', 'ece285_photos/285_pics/A56B9022.CR2', 'ece285_photos/285_pics/A56B9015.CR2', 'ece285_photos/285_pics/A56B9023.CR2']


# In[4]:


noise_dir = '285_noisy_ims'
if not os.path.isdir(noise_dir):
    os.mkdir(noise_dir)


# In[11]:


def gauss_noise(im):
    np.random.seed(0)
    gauss_im =  im + np.random.normal(scale=1, size=im.shape)
    return np.clip(gauss_im,0,1)

def quant_noise(im, fname):
    save_fp = f'{fname}.jpg'
    pil_im = Image.fromarray((im*255).astype('uint8'), 'RGB')
    pil_im.save(save_fp, quality=10, optimize=True)
    quant_im = np.asarray(Image.open(save_fp))[:,:,:]
    return np.clip(np.asarray(quant_im).astype(np.float64)/256,0,1)

def poisson_noise(im):
    np.random.seed(0)
    poiss_im = im + np.random.poisson(lam=1, size=im.shape)
    return np.clip(poiss_im,0,1)

def periodic_noise(im):
    np.random.seed(0)
    additive_freqs = np.random.choice(im.shape[1]//2+1, size=20)
    additive_freqs = np.concatenate((additive_freqs, -additive_freqs))
    if len(im.shape) == 3:
        X = np.empty_like(im)
        for i in range(3):
            fft_im = np.fft.rfft2(np.fft.fftshift(im[:, :, i]))
            fft_im[0, additive_freqs] += 1000
            X[:, :, i] = np.fft.fftshift(np.fft.irfftn(fft_im)).astype('float')
        return np.clip(X,0,1)
    else:
        fft_im = np.fft.rfft2(np.fft.fftshift(im))
        fft_im[0, additive_freqs] += 1000
        return np.clip(np.fft.fftshift(np.fft.irfftn(fft_im)).astype('float'),0,1)


# In[12]:


test = np.random.normal(size=(100, 100, 3))
# test = np.zeros((100,100, 3))
test = np.zeros((100, 100, 3), dtype='float')
test[20:70, 20:70, 0] = 1
test[20:70, 20:70, 1] = .6
test[20:70, 20:70, 2] = .1

plt.subplot(1, 2, 1)
plt.imshow(test)
plt.subplot(1, 2, 2)
# plt.imshow(quant_noise(test, 'test'))
plt.imshow(periodic_noise(test))


# In[7]:


import os
import cv2

for im_fp in pic_fnames:
    print(os.path.basename(im_fp))
    im = (np.array(Image.open(im_fp)).astype(np.float64)/256)
    im = cv2.resize(im, (600, 400))
    print(im.dtype, np.max(im))
    
    print(im_fp)
    
    np.save(f"285_noisy_ims/gt/{os.path.basename(im_fp).replace('CR2','npy')}",im)
    
    plt.imshow(im)
    plt.show()
    
    # gauss_im = gauss_noise(im)
    # quant_im = quant_noise(im, f'{im_fp}')
    # poiss_im = poisson_noise(im)
    # per_im = periodic_noise(im)
    
    # print(gauss_im.dtype, np.max(gauss_im))
    # print(quant_im.dtype, np.max(quant_im))
    # print(poiss_im.dtype, np.max(poiss_im))
    # print(per_im.dtype, np.max(per_im))
    
    #plt.figure(figsize=(100, 100))
    
    #plt.subplot(1, 4, 1)
    #plt.imshow(gauss_im)
    #
    # np.save(f"285_noisy_ims/gauss/{os.path.basename(im_fp).replace('CR2','npy')}",gauss_im)
    #plt.subplot(1, 4, 2)
    #plt.imshow(quant_im)
    # np.save(f"285_noisy_ims/quant/{os.path.basename(im_fp).replace('CR2','npy')}",quant_im)
    #plt.subplot(1, 4, 3)
    #plt.imshow(poiss_im)
    # np.save(f"285_noisy_ims/poisson/{os.path.basename(im_fp).replace('CR2','npy')}",poiss_im)
    #plt.subplot(1, 4, 4)
    #plt.imshow(per_im)
    # np.save(f"285_noisy_ims/periodic/{os.path.basename(im_fp).replace('CR2','npy')}",per_im)
    #plt.show()
    


# In[ ]:




