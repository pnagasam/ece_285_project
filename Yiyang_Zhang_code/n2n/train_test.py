import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import os 
import torchvision.transforms.functional as tvF
from model import UNet
from Config import Config as conf
import time
from dataset_processing import Training_Dataset
from torch.utils.data import Dataset, DataLoader
from dataset_processing import Testing_Dataset


if __name__ == '__main__':
    """Train the model"""
    if torch.cuda.is_available():
        print('CUDA is available. Training on GPU')
        device = torch.device('cuda')
    else:
        print('CUDA is not available. Training on CPU')
        device = torch.device('cpu')

    model = UNet(conf.img_channel, conf.img_channel)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=conf.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.MSELoss()

    dataset = Training_Dataset(conf.data_path_train, conf.gaussian_noise_param, conf.crop_img_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()


    for epoch in range(conf.max_epoch):
        for i, (x_i, y_i) in enumerate(dataloader):
            x_i = x_i.to(device)
            y_i = y_i.to(device)
            optimizer.zero_grad()
            output = model(x_i)
            loss = criterion(output, y_i)
            loss.backward()
            optimizer.step()
            if epoch % conf.save_per_epoch == 0:
                print(f'Epoch {epoch}, batch {i}, loss {loss.item()}')
        scheduler.step()
        print(f'Epoch {epoch} finished')
        # if epoch % conf.save_per_epoch == 0:
        #     save_model(model, epoch)


    # if torch.cuda.is_available():
    #     print('CUDA is available. Training on GPU')
    #     device = torch.device('cuda')
    # else:
    #     print('CUDA is not available. Training on CPU')
    #     device = torch.device('cpu')

    #load the model
    # model = UNet(in_channels=conf.img_channel,out_channels=conf.img_channel)
    # model.load_state_dict(torch.load(conf.model_path_test))
    # model.eval()
    # model.to(device)

    test_dataset = Testing_Dataset(conf.data_path_test,conf.crop_img_size)
    test_loader =  DataLoader(test_dataset, batch_size=1, shuffle=False)

    save_dir = conf.denoised_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, x_i in enumerate(test_loader):
        x_i = x_i.to(device)
        output = model(x_i).detach().cpu()

        output = tvF.to_pil_image(torch.clamp(output.squeeze(0), 0, 1))
        fname = os.path.splitext(test_loader.dataset.test_list[i])[0]
        output.save(os.path.join(save_dir, f'{fname}-denoised.png'))