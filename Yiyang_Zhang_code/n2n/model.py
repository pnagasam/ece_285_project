import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv1_to_5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv6_up5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        )
        
        self.dcoder5_up4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )

        # this is for deconvolution 4 to 2 b follow by updample 
        self.dcoder_up = nn.Sequential(
            nn.Conv2d(144, 96, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1,output_padding=1)
        )

        self.dcoderA_B_C = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1,padding=1),
            nn.LeakyReLU(0.1))
        
        self._init_weights()


    def _init_weights(self):
        """ similar to the one in homework """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        pool1 = self.conv0(x)
        pool2 = self.conv1_to_5(pool1)
        pool3 = self.conv1_to_5(pool2)
        pool4 = self.conv1_to_5(pool3)
        pool5 = self.conv1_to_5(pool4)
        

        up5 = self.conv6_up5(pool5)
        merge5 = torch.cat((up5,pool4), dim=1)
        up4 = self.dcoder5_up4(merge5)
        merge4 = torch.cat((up4,pool3), dim=1)


        up3 = self.dcoder_up(merge4)
        merge3 = torch.cat((up3, pool2), dim=1)
        # print(up3.size(), pool2.size())
        up2 = self.dcoder_up(merge3)
        merge2 = torch.cat((up2, pool1), dim=1)
        up1 = self.dcoder_up(merge2)
        merge1 = torch.cat((up1, x), dim=1)

        out = self.dcoderA_B_C(merge1)

        return out
    

