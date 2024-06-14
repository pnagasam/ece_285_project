#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import os


# In[6]:


noisy_im_dir = '285_noisy_ims'


# In[14]:


noisy_ims = {}

for noise_type_dir in os.listdir(noisy_im_dir):
    curr_pth = os.path.join(noisy_im_dir, noise_type_dir)
    for pth in os.listdir(curr_pth):
        print(pth)
        if (pth.endswith('A56B9022.npy') or pth.endswith('A56B9043.npy')) or pth.endswith('A56B9023.npy'):
            im_pth = os.path.join(curr_pth, pth)
            im = np.load(im_pth)
            
            plt.imshow(im)
            plt.show()
            
            if noise_type_dir in noisy_ims.keys():
                noisy_ims[noise_type_dir].append(im)
            else:
                noisy_ims[noise_type_dir] = [im]


# In[8]:


import bm3d
import cv2


# In[16]:


noisy_ims


# In[23]:


from PIL import Image


# In[ ]:


pic_fnames = ['ece285_photos/out_door_pics/A56B9042.CR2', 'ece285_photos/out_door_pics/A56B9043.CR2', 'ece285_photos/285_pics/A56B9022.CR2', 'ece285_photos/285_pics/A56B9015.CR2', 'ece285_photos/285_pics/A56B9023.CR2']


# In[25]:


ground_truth = ['A56B9022.npy', 'A56B9043.npy', 'A56B9023.npy']


# In[ ]:


pth.endswith('A56B9022.npy') or pth.endswith('A56B9043.npy')) or pth.endswith('A56B9023.npy')


# denoised_ims = {}

# In[27]:


denoised_ims = {}

for key in noisy_ims.keys():
    noise_type_ims = noisy_ims[key]
    denoised_ims[key] = []
    
    print(len(noise_type_ims))
    
    for i in range(len(noise_type_ims)):
        
        gt_im = np.load(os.path.join(noisy_im_dir, 'gt', ground_truth[i]))
        
        noisy_im = noise_type_ims[i]
        
        denoised_im = np.empty_like(noisy_im)
        
        if key == 'gaussian':
            denoised_im = bm3d.bm3d(noisy_im, sigma_psd=.1, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            denoised_ims[key].append(denoised_im)
        
        else:
            denoised_im = bm3d.bm3d(noisy_im, sigma_psd=.3, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            denoised_ims[key].append(denoised_im)
            
        
        psnr = cv2.PSNR(gt_im, denoised_im)
        
        print(psnr)
        
        plt.figure(figsize=(100, 100))
        plt.subplot(1,3,3)
        plt.imshow(gt_im)
        plt.subplot(1,3,1)
        plt.imshow(noisy_im)
        plt.subplot(1,3,2)
        plt.imshow(denoised_im)
        plt.title(psnr)
        plt.show()
        
        cv2.imwrite(f"bm3d_denoised/{i}_{key}.png",(denoised_im*255).astype('uint8'))


# In[ ]:





# In[ ]:





# In[30]:


from skimage.restoration import denoise_tv_chambolle


# In[31]:


denoised_ims_tv = {}

for key in noisy_ims.keys():
    noise_type_ims = noisy_ims[key]
    denoised_ims_tv[key] = []
    print(len(noise_type_ims))
    print(key)
    for i in range(len(noise_type_ims)):
        
        gt_im = np.load(os.path.join(noisy_im_dir, 'gt', ground_truth[i]))
        
        noisy_im = noise_type_ims[i]
        
        denoised_im = np.empty_like(noisy_im)
        
        if key == 'poisson':
            denoised_im = denoise_tv_chambolle(noisy_im, weight=.5, channel_axis=2)
            denoised_ims_tv[key].append(denoised_im)
            
        else:
            denoised_im = denoise_tv_chambolle(noisy_im, weight=.3, channel_axis=2)
            denoised_ims_tv[key].append(denoised_im)
        
        
        psnr = cv2.PSNR(gt_im, denoised_im)
        
        print(psnr)
        
        plt.figure(figsize=(100, 100))
        plt.subplot(1,3,3)
        plt.imshow(gt_im)
        plt.subplot(1,3,1)
        plt.imshow(noisy_im)
        plt.subplot(1,3,2)
        plt.imshow(denoised_im)
        plt.title(psnr)
        plt.show()
        
        cv2.imwrite(f"tv_denoised/{i}_{key}.png",(denoised_im*255).astype('uint8'))


# In[29]:


denoised_ims_periodic = []

for i in range(len(noisy_ims['periodic'])):
        
        gt_im = np.load(os.path.join(noisy_im_dir, 'gt', ground_truth[i]))
        
        noisy_im = noisy_ims['periodic'][i]
        
        denoised_im = np.empty_like(noisy_im)
        
        max_peaks =  [ 172,   47, -172,  -47]
        for i in range(3):
                
                fft_im = np.fft.rfft2(np.fft.fftshift(noisy_im[:, :, i]))
        
                for max_i in max_peaks:
                        fft_im[0, max_i] = fft_im[-2, max_i-2]
                
                denoised_im[:, :, i] = np.fft.fftshift(np.fft.irfftn(fft_im))

        
        psnr = cv2.PSNR(gt_im, denoised_im)
        
        print(psnr)
        
        plt.figure(figsize=(100, 100))
        plt.subplot(1,3,3)
        plt.imshow(gt_im)
        plt.subplot(1,3,1)
        plt.imshow(noisy_im)
        plt.subplot(1,3,2)
        plt.imshow(denoised_im)
        plt.title(psnr)
        plt.show()
        
        cv2.imwrite(f"periodic_denoised/{i}_{key}.png",(denoised_im*255).astype('uint8'))


# In[8]:


noisy_im_periodic = noisy_ims['periodic'][0]


# In[13]:


# plt.imshow(noisy_im_periodic)
import cv2

denoised_im = np.empty_like(noisy_im_periodic)
max_peaks =  [ 172,   47, -172,  -47]
for i in range(3):
    fft_im = np.fft.rfft2(np.fft.fftshift(noisy_im_periodic[:, :, i]))
    plot_im = fft_im.copy()
    plot_im[0,0] = 0
    plt.imshow(cv2.dilate(np.log(abs(np.fft.fftshift(plot_im, axes=0))), kernel=np.ones((10, 10))))
    plt.show()
    
    for max_i in max_peaks:
        fft_im[0, max_i] = fft_im[-2, max_i-2]
    
    plot_im = fft_im.copy()
    plot_im[0,0] = 0
    plt.imshow(np.log(abs(np.fft.fftshift(plot_im, axes=0))))
    plt.show()
    denoised_im[:, :, i] = np.fft.fftshift(np.fft.irfftn(fft_im))

plt.figure(figsize=(100, 100))
plt.subplot(1,2,1)
plt.imshow(noisy_im_periodic)
plt.subplot(1,2,2)
plt.imshow(np.real(denoised_im))
plt.show()


# In[17]:


ground_truth = ['ece285_photos/out_door_pics/A56B9042.CR2', 'ece285_photos/out_door_pics/A56B9043.CR2', 'ece285_photos/285_pics/A56B9022.CR2', 'ece285_photos/285_pics/A56B9015.CR2', 'ece285_photos/285_pics/A56B9023.CR2']


# In[18]:


for key in denoised_ims.keys():
    plt.figure(figsize=(8, 8))
    plt.imshow(denoised_ims[key])
    plt.title(key)
    plt.show()


# In[37]:


import matplotlib.pyplot as plt


# In[52]:


# periodic_denoised/{i}_{key}.png
# from matplotlib.pyplot import savefig

def plot(im_ind, method):
    
    gt_im = np.load(os.path.join(noisy_im_dir, 'gt', ground_truth[im_ind]))
    
    plt.imshow(gt_im)
    plt.show()
    
    plt.figure(figsize=(8,8))
    i = 1
    for key in noisy_ims.keys():
        plt.subplot(2,2,i)
        im = cv2.imread(f'{method}_denoised/{im_ind}_{key}.png', -1)
        plt.imshow(im)
        plt.axis('off')
        plt.title(key)
        i+=1
    
    
    plt.savefig(f'{im_ind}_{method}.png', transparent=True, bbox_inches='tight', pad_inches = 0)
    plt.show()
        


# In[59]:


plot(1, 'tv')


# In[ ]:




