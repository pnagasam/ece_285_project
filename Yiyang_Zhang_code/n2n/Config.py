"""

from github noise2noise_model_for_natural_img/unet_model.py

"""

class Config:
    data_path_train = 'data/train'
    data_path_test = 'data/test'
    data_path_checkpoint = "data/checkpoint/dpc"
    model_path_test= "data/checkpoint/dpc/n2n-epoch100.pth"
    denoised_dir = 'data/result'
    img_channel = 3
    max_epoch = 10
    crop_img_size = 256
    learning_rate = 0.001
    save_per_epoch = 5
    gaussian_noise_param = 30
    test_noise_param = 70
    cuda = "cuda:1"